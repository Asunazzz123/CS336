#!/usr/bin/env bash
# Run in the foreground; wrap the whole pipeline in nohup when using SSH.
set -e -o pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
LAUNCHER="$SCRIPT_DIR/run_training.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'EOF'
Usage: bash scripts/start_h20_pipeline.sh
Runs one GSM8K SFT epoch, then 100 RL updates; waits for SFT success first.
Default: Qwen3.5-9B-Base-text, GPU0 training, GPU1 vLLM, CPU AdamW state.
Use nohup for the entire pipeline (not just SFT):
  mkdir -p outputs/pipelines
  nohup bash scripts/start_h20_pipeline.sh >outputs/pipelines/launch.log 2>&1 </dev/null &
Stage control: bash scripts/run_training.sh status|logs|stop <absolute stage directory>
Stopping the active stage also fails this pipeline and prevents the next stage.
Overrides: MODEL_PATH, TRAIN_PATH, PIPELINE_ROOT, PIPELINE_NAME, MIN_FREE_GIB,
  SFT_EPOCHS, SFT_MAX_STEPS (unset = full epoch), RL_STEPS, RL_MODE (grpo|grpo-full),
  GENERATION_BACKEND (vllm|transformers), GENERATION_BATCH_SIZE,
  ADVANTAGE_NORMALIZER, LOSS_NORMALIZATION.
No optimizer state is saved; checkpoints support inference and stage handoff.
EOF
    exit 0
fi
[[ $# == 0 ]] || { printf 'Unexpected arguments; use --help.\n' >&2; exit 2; }
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3.5-9B-Base-text}
TRAIN_PATH=${TRAIN_PATH:-"$PROJECT_DIR/data/gsm8k/train.jsonl"}
PIPELINE_ROOT=${PIPELINE_ROOT:-"$PROJECT_DIR/outputs/pipelines"}
PIPELINE_NAME=${PIPELINE_NAME:-"h20-gsm8k-$(date '+%Y%m%d-%H%M%S')-$$"}
RL_MODE=${RL_MODE:-grpo}
GENERATION_BACKEND=${GENERATION_BACKEND:-vllm}
if [[ "$GENERATION_BACKEND" == vllm ]]; then default_generation_batch=8
else default_generation_batch=1; fi
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-$default_generation_batch}
MIN_FREE_GIB=${MIN_FREE_GIB:-45}
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
[[ "$PIPELINE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail 'Invalid PIPELINE_NAME'
[[ "$RL_MODE" == grpo || "$RL_MODE" == grpo-full ]] || fail 'Invalid RL_MODE'
[[ "$GENERATION_BACKEND" == vllm || "$GENERATION_BACKEND" == transformers ]] || fail 'Invalid GENERATION_BACKEND'
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] || fail 'MIN_FREE_GIB must be a nonnegative integer'
[[ -f "$TRAIN_PATH" ]] || fail "Training data missing: $TRAIN_PATH"
[[ -f "$MODEL_PATH/config.json" ]] || fail "Model config missing: $MODEL_PATH"
[[ -f "$MODEL_PATH/model.safetensors" || -f "$MODEL_PATH/model.safetensors.index.json" ]] || fail "Model weights missing: $MODEL_PATH"
MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd -P)
TRAIN_PATH="$(cd -- "$(dirname -- "$TRAIN_PATH")" && pwd -P)/$(basename -- "$TRAIN_PATH")"
mkdir -p -- "$PIPELINE_ROOT"
PIPELINE_ROOT=$(cd -- "$PIPELINE_ROOT" && pwd -P)
free_kib=$(df -Pk "$PIPELINE_ROOT" | awk 'END {print $4}')
((free_kib >= MIN_FREE_GIB * 1024 * 1024)) || fail "Need at least $MIN_FREE_GIB GiB free for SFT and RL model exports"
PIPELINE_DIR="$PIPELINE_ROOT/$PIPELINE_NAME"
mkdir -- "$PIPELINE_DIR" || fail "Pipeline already exists: $PIPELINE_DIR"
mkdir -- "$PIPELINE_DIR/stages"
ln -sfn -- "$PIPELINE_DIR" "$PIPELINE_ROOT/latest"
stage_job=''
stage_pid=''
stopping=0
finish() {
    code=$?
    trap - EXIT
    state=failed
    if [[ "$stopping" == 1 ]]; then state=stopped
    elif [[ "$code" == 0 ]]; then state=succeeded; fi
    printf '%s\n' "$state" > "$PIPELINE_DIR/status"
    printf '%s\n' "$code" > "$PIPELINE_DIR/exit_code"
    printf 'Pipeline %s; exit_code=%s; directory=%s\n' "$state" "$code" "$PIPELINE_DIR"
    exit "$code"
}
stop_stage() {
    stopping=1
    trap '' TERM INT
    if [[ -n "$stage_pid" ]]; then
        for attempt in {1..30}; do
            [[ -f "$stage_job/pid" || -f "$stage_job/exit_code" ]] && break
            kill -0 "$stage_pid" 2>/dev/null || break
            sleep 0.1
        done
        if [[ -f "$stage_job/pid" ]]; then bash "$LAUNCHER" stop "$stage_job" || true; fi
        wait "$stage_pid" 2>/dev/null || true
    fi
    exit 143
}
trap finish EXIT
trap stop_stage TERM INT
trap '' HUP
printf 'running\n' > "$PIPELINE_DIR/status"
printf '%s\n' "$$" > "$PIPELINE_DIR/pid"
printf 'Pipeline: %s\nSFT: %s epoch(s); RL: %s updates (not a full dataset pass)\n' \
    "$PIPELINE_DIR" "${SFT_EPOCHS:-1}" "${RL_STEPS:-100}"
run_stage() {
    local stage="$1" mode="$2" code
    shift 2
    stage_job="$PIPELINE_DIR/stages/$stage"
    printf '%s\n' "$stage" > "$PIPELINE_DIR/stage"
    printf '%s\n' "$stage_job" > "$PIPELINE_DIR/current_stage"
    RUN_ROOT="$PIPELINE_DIR/stages" RUN_NAME="$stage" bash "$LAUNCHER" run "$mode" "$@" &
    stage_pid=$!
    if wait "$stage_pid"; then code=0; else code=$?; fi
    stage_pid=''
    return "$code"
}
# FP32 parameters and CPU optimizer state; BF16 autocast on the training GPU.
common=(--device cuda:0 --dtype float32 --autocast bfloat16 --gradient-checkpointing
        --optimizer-name adamw --optimizer-device cpu --no-save-optimizer-state
        --save-every 2147483647 --eval-examples 64)
sft_limit=()
if [[ -n "${SFT_MAX_STEPS:-}" ]]; then sft_limit=(--max-steps "$SFT_MAX_STEPS"); fi
run_stage sft sft "${common[@]}" --model-id "$MODEL_PATH" --train-path "$TRAIN_PATH" \
    --num-epochs "${SFT_EPOCHS:-1}" --batch-size 1 --gradient-accumulation-steps 8 \
    --max-seq-length 1024 --learning-rate 1e-5 --eval-every 50 "${sft_limit[@]}"
shopt -s nullglob
checkpoints=("$PIPELINE_DIR/stages/sft/artifacts"/checkpoint-*)
[[ ${#checkpoints[@]} == 1 && -d "${checkpoints[0]}" ]] || fail 'Expected exactly one final SFT checkpoint'
checkpoint=${checkpoints[0]}
[[ -f "$checkpoint/model.safetensors" || -f "$checkpoint/model.safetensors.index.json" ]] || fail 'SFT checkpoint has no model weights'
printf '%s\n' "$checkpoint" > "$PIPELINE_DIR/sft_checkpoint"
run_stage rl "$RL_MODE" "${common[@]}" --model-id "$checkpoint" \
    --train-path "$PIPELINE_DIR/stages/sft/artifacts/data/train.jsonl" \
    --eval-path "$PIPELINE_DIR/stages/sft/artifacts/data/validation.jsonl" \
    --num-steps "${RL_STEPS:-100}" --prompts-per-batch 2 --group-size 4 \
    --gradient-accumulation-steps 8 --max-new-tokens 512 --learning-rate 1e-6 \
    --eval-every 20 --generation-batch-size "$GENERATION_BATCH_SIZE" --generation-backend "$GENERATION_BACKEND" \
    --vllm-gpu 1 --advantage-normalizer "${ADVANTAGE_NORMALIZER:-none}" \
    --loss-normalization "${LOSS_NORMALIZATION:-constant}"
