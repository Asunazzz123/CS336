#!/usr/bin/env bash
# Start with `bash scripts/run_training.sh [start] [grpo-full|grpo|sft] [trainer args]`.
set -e -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
SCRIPT_PATH="$SCRIPT_DIR/run_training.sh"
ENTRY_DIR="$PROJECT_DIR/scripts"
RUN_ROOT=${RUN_ROOT:-"$PROJECT_DIR/outputs/jobs"}
case "$RUN_ROOT" in /*) ;; *) RUN_ROOT="$PROJECT_DIR/$RUN_ROOT" ;; esac
CONDA_ENV=${CONDA_ENV:-agent}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_training.sh [start] [grpo-full|grpo|sft] [trainer arguments...]
  bash scripts/run_training.sh run [grpo-full|grpo|sft] [trainer arguments...]
  bash scripts/run_training.sh status [JOB_NAME|JOB_DIR]
  bash scripts/run_training.sh logs [JOB_NAME|JOB_DIR] [-f|--follow]
  bash scripts/run_training.sh stop [JOB_NAME|JOB_DIR]

No arguments: start grpo-full in the background; SSH can disconnect afterwards.
run: stay in the foreground for debugging. status/logs/stop default to latest.
Defaults: Qwen/Qwen3.5-4B-Base, cuda:0, bfloat16, CONDA_ENV=agent.
All trainer flags are forwarded literally and override these defaults.
SFT checkpoints must be passed explicitly with --model-id for subsequent RL.

Environment:
  CONDA_EXE  Conda executable (automatically discovered when unset)
  CONDA_ENV  Conda environment name (default: agent)
  RUN_ROOT   Job control/log directory (default: assignment5/outputs/jobs)
  RUN_NAME   Unique job directory name (default: mode + timestamp + launcher PID)
  TRAINING_BACKEND  transformers (default) or fsdp2; also accepts --training-backend
  NPROC_PER_NODE    FSDP ranks on this host (default: 2)

Each job contains train.log, command.sh, pid, pid_start, status and exit_code.
Trainer --output-dir defaults to JOB_DIR/artifacts, separate from control files.
EOF
}

resolve_conda() {
    local candidate
    if [[ -n "${CONDA_EXE:-}" ]]; then
        if [[ -x "$CONDA_EXE" ]]; then
            CONDA_BIN="$(cd -- "$(dirname -- "$CONDA_EXE")" && pwd -P)/$(basename -- "$CONDA_EXE")"
            return
        fi
        candidate=$(command -v -- "$CONDA_EXE" 2>/dev/null || true)
        [[ -n "$candidate" && -x "$candidate" ]] || fail "CONDA_EXE is not an executable: $CONDA_EXE"
        CONDA_BIN="$candidate"
        return
    fi
    candidate=$(type -P conda 2>/dev/null || true)
    if [[ -n "$candidate" ]]; then CONDA_BIN="$candidate"; return; fi
    for candidate in "${HOME:-}/miniconda3/bin/conda" "${HOME:-}/anaconda3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; return; fi
    done
    fail "Conda not found. Set CONDA_EXE to its absolute executable path."
}

has_trainer_flag() {
    local wanted="$1" argument
    for argument in "${TRAIN_ARGS[@]}"; do
        case "$argument" in "$wanted"|"$wanted"=*) return 0 ;; esac
    done
    return 1
}

add_default() {
    if ! has_trainer_flag "$1"; then COMMAND+=("$1" "$2"); fi
}

make_command() {
    local job="$1" mode="$2" entry backend="${TRAINING_BACKEND:-transformers}" previous='' argument
    shift 2
    TRAIN_ARGS=("$@")
    for argument in "${TRAIN_ARGS[@]}"; do
        if [[ "$previous" == --training-backend ]]; then backend="$argument"; fi
        case "$argument" in --training-backend=*) backend="${argument#*=}" ;; esac
        previous="$argument"
    done
    [[ "$backend" == transformers || "$backend" == fsdp2 ]] || fail "Unknown training backend: $backend"
    case "$mode" in
        sft) entry=train_sft.py ;;
        grpo|grpo-full) entry=train_grpo_unified.py ;;
        *) fail "Unknown training mode: $mode" ;;
    esac
    resolve_conda
    if [[ "$backend" == fsdp2 ]]; then
        [[ "${NPROC_PER_NODE:-2}" =~ ^[0-9]+$ && "${NPROC_PER_NODE:-2}" -ge 2 ]] || fail 'FSDP needs NPROC_PER_NODE >=2'
        if [[ "$mode" != sft ]]; then entry=train_grpo_unified.py; fi
        COMMAND=("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -u -m torch.distributed.run
                 --standalone "--nproc-per-node=${NPROC_PER_NODE:-2}" "$ENTRY_DIR/$entry")
        add_default --training-backend fsdp2
    else
        COMMAND=("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -u "$ENTRY_DIR/$entry")
    fi
    if [[ "$mode" == grpo ]]; then
        add_default --num-epochs-per-rollout 1
        add_default --advantage-normalizer none
        add_default --loss-normalization constant
    elif [[ "$mode" == grpo-full ]]; then
        add_default --num-epochs-per-rollout 3
    fi
    add_default --model-id Qwen/Qwen3.5-4B-Base
    add_default --device cuda:0
    add_default --dtype bfloat16
    add_default --output-dir "$job/artifacts"
    COMMAND+=("${TRAIN_ARGS[@]}")
}

write_value() {
    local target="$1" value="$2"
    printf '%s\n' "$value" > "$target.tmp.$$"
    mv -f -- "$target.tmp.$$" "$target"
}

process_start() { TZ=UTC LC_ALL=C ps -p "$1" -o lstart= 2>/dev/null | sed 's/^ *//' || true; }

owned_worker() {
    local job="$1" pid stamp command_line
    [[ -f "$job/pid" && -f "$job/pid_start" ]] || return 1
    pid=$(cat -- "$job/pid")
    [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    stamp=$(process_start "$pid")
    [[ -n "$stamp" && "$stamp" == "$(cat -- "$job/pid_start")" ]] || return 1
    command_line=$(LC_ALL=C ps -ww -p "$pid" -o args= 2>/dev/null || true)
    case "$command_line" in
        *"$SCRIPT_PATH"*"__worker"*"$job"*|*"$PROJECT_DIR/scripts/run_training.sh"*"__worker"*"$job"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Keep identities after a parent exits: a stubborn Python/vLLM child can be reparented.
snapshot_tree() {
    local parent="$1" stamp child
    stamp=$(process_start "$parent")
    [[ -n "$stamp" ]] || return 0
    for child in $(ps -e -o pid= -o ppid= | awk -v parent="$parent" '$2 == parent {print $1}'); do
        snapshot_tree "$child"
    done
    STOP_PIDS+=("$parent")
    STOP_STARTS+=("$stamp")
}

snapshot_member_alive() {
    local index="$1" pid state
    pid="${STOP_PIDS[$index]}"
    [[ "${STOP_STARTS[$index]}" == "$(process_start "$pid")" ]] || return 1
    state=$(ps -p "$pid" -o stat= 2>/dev/null | tr -d ' ' || true)
    [[ -n "$state" && "$state" != Z* ]] && kill -0 "$pid" 2>/dev/null
}

signal_snapshot() {
    local signal="$1" index
    for ((index=0; index<${#STOP_PIDS[@]}; index++)); do
        if snapshot_member_alive "$index"; then
            kill -s "$signal" "${STOP_PIDS[$index]}" 2>/dev/null || true
        fi
    done
}

terminate_children() {
    local parent="$1" attempt index alive
    STOP_PIDS=()
    STOP_STARTS=()
    snapshot_tree "$parent"
    signal_snapshot TERM
    for attempt in {1..20}; do
        alive=0
        for ((index=0; index<${#STOP_PIDS[@]}; index++)); do
            if snapshot_member_alive "$index"; then alive=1; break; fi
        done
        if [[ "$alive" == 0 ]]; then return; fi
        sleep 0.1
    done
    signal_snapshot KILL
}

worker() {
    local job="$1" mode="$2" child_pid='' stopping=0 code
    shift 2
    trap '' HUP
    finish_worker() {
        local result=$? state=failed
        trap - EXIT
        if [[ "$stopping" == 1 ]]; then state=stopped
        elif [[ "$result" == 0 ]]; then state=succeeded; fi
        write_value "$job/status" "$state"
        write_value "$job/exit_code" "$result"
        write_value "$job/finished_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf 'Training %s; exit_code=%s\n' "$state" "$result"
        exit "$result"
    }
    stop_worker() {
        stopping=1
        trap '' TERM INT
        # $! also covers TERM arriving between spawn and child_pid assignment.
        if [[ -z "$child_pid" ]]; then child_pid=$!; fi
        if [[ -n "$child_pid" ]]; then
            terminate_children "$child_pid"
            wait "$child_pid" 2>/dev/null || true
        fi
        exit 143
    }
    trap finish_worker EXIT
    trap stop_worker TERM INT
    cd -- "$PROJECT_DIR"
    make_command "$job" "$mode" "$@"
    local stamp
    stamp=$(process_start "$$")
    [[ -n "$stamp" ]] || fail "Cannot inspect worker identity with ps"
    write_value "$job/pid_start" "$stamp"
    write_value "$job/pid" "$$"
    write_value "$job/status" running
    printf 'Project: %s\nMode: %s\nConda environment: %s\n' "$PROJECT_DIR" "$mode" "$CONDA_ENV"
    printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
    "${COMMAND[@]}" &
    child_pid=$!
    write_value "$job/child_pid" "$child_pid"
    if wait "$child_pid"; then code=0; else code=$?; fi
    exit "$code"
}

prepare_job() {
    local mode="$1" name
    name=${RUN_NAME:-"${mode}-$(date '+%Y%m%d-%H%M%S')-$$"}
    [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "RUN_NAME must be a simple unique job name"
    [[ -n "$(process_start "$$")" ]] || fail "Cannot inspect processes with ps; process visibility is required"
    mkdir -p -- "$RUN_ROOT"
    RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd -P)
    JOB_DIR="$RUN_ROOT/$name"
    # Preflight the command before creating job files; no Python/model is launched.
    make_command "$JOB_DIR" "$mode" "${TRAIN_ARGS[@]}"
    mkdir -- "$JOB_DIR" 2>/dev/null || fail "Job already exists: $JOB_DIR (choose a new RUN_NAME)"
    write_value "$JOB_DIR/status" starting
    write_value "$JOB_DIR/mode" "$mode"
    write_value "$JOB_DIR/started_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    { printf '#!/usr/bin/env bash\ncd -- %q\n' "$PROJECT_DIR"; printf '%q ' "${COMMAND[@]}"; printf '\n'; } > "$JOB_DIR/command.sh"
    if [[ -e "$RUN_ROOT/latest" && ! -L "$RUN_ROOT/latest" ]]; then
        fail "$RUN_ROOT/latest exists and is not a symlink"
    fi
    ln -sfn -- "$JOB_DIR" "$RUN_ROOT/latest"
}

resolve_job() {
    local reference="${1:-latest}" candidate
    case "$reference" in
        /*) candidate="$reference" ;;
        *)
            if [[ -d "$reference" ]]; then candidate="$reference"
            else candidate="$RUN_ROOT/$reference"; fi ;;
    esac
    [[ -d "$candidate" ]] || fail "Job not found: $reference"
    JOB_DIR=$(cd -- "$candidate" && pwd -P)
    [[ -f "$JOB_DIR/command.sh" && -f "$JOB_DIR/status" ]] || fail "Not a launcher job: $JOB_DIR"
}

show_status() {
    local state
    state=$(cat -- "$JOB_DIR/status")
    if [[ "$state" == running || "$state" == starting ]]; then
        if ! owned_worker "$JOB_DIR"; then state='not-running (no exit status recorded)'; fi
    fi
    printf 'Job: %s\nStatus: %s\n' "$JOB_DIR" "$state"
    if [[ -f "$JOB_DIR/pid" ]]; then printf 'PID: %s\n' "$(cat -- "$JOB_DIR/pid")"; fi
    if [[ -f "$JOB_DIR/exit_code" ]]; then printf 'Exit code: %s\n' "$(cat -- "$JOB_DIR/exit_code")"; fi
    printf 'Log: %s/train.log\n' "$JOB_DIR"
}

stop_job() {
    local pid attempt
    if ! owned_worker "$JOB_DIR"; then
        if [[ -f "$JOB_DIR/exit_code" ]]; then show_status; return; fi
        fail "Worker identity could not be verified; refusing to signal a possibly reused PID"
    fi
    pid=$(cat -- "$JOB_DIR/pid")
    kill -TERM "$pid"
    for attempt in {1..100}; do
        if [[ -f "$JOB_DIR/exit_code" ]]; then show_status; return; fi
        sleep 0.1
    done
    # Snapshot/terminate the still-owned process tree, never a broad process-name match.
    if owned_worker "$JOB_DIR"; then
        STOP_PIDS=()
        STOP_STARTS=()
        snapshot_tree "$pid"
        signal_snapshot KILL
        write_value "$JOB_DIR/status" stopped
        write_value "$JOB_DIR/exit_code" 137
        write_value "$JOB_DIR/finished_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    fi
    show_status
}

main() {
    local action=start mode=grpo-full attempt reference='' follow=0
    case "${1:-}" in
        __worker) shift; worker "$@"; return ;;
        help|--help|-h) usage; return ;;
        start|run|status|logs|stop) action="$1"; shift ;;
        sft|grpo|grpo-full|'') ;;
        --*) ;;
        *) fail "Unknown command: $1 (use --help)" ;;
    esac
    case "$action" in
        start|run)
            case "${1:-}" in
                sft|grpo|grpo-full) mode="$1"; shift ;;
                --*|'') ;;
                *) fail "Unknown training mode: $1" ;;
            esac
            TRAIN_ARGS=("$@")
            prepare_job "$mode"
            printf 'Job: %s\nLog: %s/train.log\n' "$JOB_DIR" "$JOB_DIR"
            if [[ "$action" == run ]]; then
                bash "$SCRIPT_PATH" __worker "$JOB_DIR" "$mode" "$@" 2>&1 | tee -a "$JOB_DIR/train.log"
                return
            fi
            trap '' HUP
            if command -v setsid >/dev/null 2>&1; then
                nohup setsid bash "$SCRIPT_PATH" __worker "$JOB_DIR" "$mode" "$@" </dev/null >>"$JOB_DIR/train.log" 2>&1 &
            else
                nohup bash "$SCRIPT_PATH" __worker "$JOB_DIR" "$mode" "$@" </dev/null >>"$JOB_DIR/train.log" 2>&1 &
            fi
            for attempt in {1..50}; do
                if [[ -f "$JOB_DIR/pid" || -f "$JOB_DIR/exit_code" ]]; then break; fi
                sleep 0.1
            done
            [[ -f "$JOB_DIR/pid" || -f "$JOB_DIR/exit_code" ]] || fail "Worker did not start; inspect $JOB_DIR/train.log"
            printf 'Started in background. SSH can disconnect.\n'
            if [[ -f "$JOB_DIR/pid" ]]; then printf 'PID: %s\n' "$(cat -- "$JOB_DIR/pid")"; fi
            printf 'Status: bash %q status %q\n' "$SCRIPT_PATH" "$JOB_DIR"
            ;;
        status|stop)
            [[ $# -le 1 ]] || fail "$action accepts at most one job reference"
            resolve_job "${1:-latest}"
            if [[ "$action" == status ]]; then show_status; else stop_job; fi ;;
        logs)
            for argument in "$@"; do
                case "$argument" in
                    -f|--follow) follow=1 ;;
                    *) [[ -z "$reference" ]] || fail "logs accepts one job reference"; reference="$argument" ;;
                esac
            done
            resolve_job "${reference:-latest}"
            if [[ "$follow" == 1 ]]; then tail -n 100 -f "$JOB_DIR/train.log"
            else tail -n 100 "$JOB_DIR/train.log"; fi ;;
    esac
}

main "$@"
