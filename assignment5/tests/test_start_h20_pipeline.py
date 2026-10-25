"""Check stage handoff and SSH lifecycle with a fake Conda executable."""

import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


PROJECT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT / "scripts/start_h20_pipeline.sh"
LAUNCHER = PROJECT / "scripts/run_training.sh"


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("Pipeline did not reach the expected state")


@pytest.fixture
def pipeline_env(tmp_path):
    model = tmp_path / "model with spaces"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    data = tmp_path / "gsm8k.jsonl"
    data.touch()
    conda = tmp_path / "fake conda"
    conda.write_text("""#!/usr/bin/env bash
set -eu
stage=''
output=''
previous=''
for argument in "$@"; do
    case "$argument" in
        */train_sft.py) stage=sft ;;
        */train_grpo.py|*/train_grpo_full.py) stage=rl ;;
    esac
    if [[ "$previous" == --output-dir ]]; then output="$argument"; fi
    previous="$argument"
done
printf '%s\\n' "$@" > "$RECORD_ROOT/$stage.args"
if [[ "$stage" == sft ]]; then
    if [[ "${SFT_FAIL:-0}" != 0 ]]; then exit "$SFT_FAIL"; fi
    if [[ -n "${SFT_RELEASE:-}" ]]; then
        while [[ ! -f "$SFT_RELEASE" ]]; do sleep 0.05; done
    fi
    mkdir -p "$output/data" "$output/checkpoint-000841"
    touch "$output/data/train.jsonl" "$output/data/validation.jsonl"
    touch "$output/checkpoint-000841/model.safetensors.index.json"
    touch "$RECORD_ROOT/sft.finished"
else
    [[ -f "$RECORD_ROOT/sft.finished" ]] || exit 99
fi
""", encoding="utf-8")
    conda.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"], "CONDA_EXE": str(conda), "CONDA_ENV": "agent",
        "MODEL_PATH": str(model), "TRAIN_PATH": str(data), "MIN_FREE_GIB": "0",
        "PIPELINE_ROOT": str(tmp_path / "pipelines"), "PIPELINE_NAME": "test-pipeline",
        "RECORD_ROOT": str(tmp_path),
    }
    yield env
    job = Path(env["PIPELINE_ROOT"]) / env["PIPELINE_NAME"]
    active = job / "current_stage"
    if active.exists():
        subprocess.run(["bash", str(LAUNCHER), "stop", active.read_text().strip()],
                       env=env, capture_output=True, timeout=15, check=False)
    if "SFT_RELEASE" in env:
        Path(env["SFT_RELEASE"]).touch()


def pipeline_dir(env):
    return Path(env["PIPELINE_ROOT"]) / env["PIPELINE_NAME"]


def args_for(env, stage):
    return (Path(env["RECORD_ROOT"]) / f"{stage}.args").read_text().splitlines()


def test_sft_then_rl_handoff_and_full_epoch_defaults(pipeline_env):
    result = subprocess.run(["bash", str(PIPELINE)], env=pipeline_env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    sft, rl = args_for(pipeline_env, "sft"), args_for(pipeline_env, "rl")
    assert sft[sft.index("--model-id") + 1] == pipeline_env["MODEL_PATH"]
    assert "--max-steps" not in sft
    assert sft[sft.index("--num-epochs") + 1] == "1"
    assert rl[rl.index("--num-steps") + 1] == "100"
    assert rl[rl.index("--generation-batch-size") + 1] == "8"
    assert rl[rl.index("--advantage-normalizer") + 1] == "none"
    assert rl[rl.index("--loss-normalization") + 1] == "constant"
    checkpoint = pipeline_dir(pipeline_env) / "stages/sft/artifacts/checkpoint-000841"
    assert rl[rl.index("--model-id") + 1] == str(checkpoint)
    assert rl[rl.index("--train-path") + 1] == str(checkpoint.parent / "data/train.jsonl")
    for args in (sft, rl):
        assert args[args.index("--optimizer-device") + 1] == "cpu"
        assert args[args.index("--dtype") + 1] == "float32"
        assert args[args.index("--autocast") + 1] == "bfloat16"
        assert args[args.index("--save-every") + 1] == "2147483647"
        assert "--no-save-optimizer-state" in args
        assert "--gradient-checkpointing" in args
    assert any(arg.endswith("/train_grpo.py") for arg in rl)
    assert (pipeline_dir(pipeline_env) / "status").read_text().strip() == "succeeded"


def test_sft_failure_prevents_rl(pipeline_env):
    pipeline_env["SFT_FAIL"] = "23"
    result = subprocess.run(["bash", str(PIPELINE)], env=pipeline_env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 23, result.stdout + result.stderr
    assert not (Path(pipeline_env["RECORD_ROOT"]) / "rl.args").exists()
    assert (pipeline_dir(pipeline_env) / "status").read_text().strip() == "failed"


def test_whole_pipeline_survives_launching_shell_hangup(pipeline_env):
    record = Path(pipeline_env["RECORD_ROOT"])
    release, ready = record / "release", record / "shell.ready"
    pipeline_env["SFT_RELEASE"] = str(release)
    shell = subprocess.Popen(
        ["bash", "-c", 'nohup bash "$1" >"$2" 2>&1 </dev/null & '
         'touch "$3"; while :; do sleep 0.1; done',
         "launch-shell", str(PIPELINE), str(record / "pipeline.log"), str(ready)],
        env=pipeline_env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(lambda: ready.exists() and (record / "sft.args").exists())
        assert not (record / "rl.args").exists()
        os.killpg(shell.pid, signal.SIGHUP)
        shell.wait(timeout=5)
        assert not (pipeline_dir(pipeline_env) / "exit_code").exists()
        release.touch()
        wait_for(lambda: (pipeline_dir(pipeline_env) / "exit_code").exists())
        assert (pipeline_dir(pipeline_env) / "status").read_text().strip() == "succeeded"
        assert (record / "rl.args").exists()
    finally:
        release.touch()
        if shell.poll() is None:
            os.killpg(shell.pid, signal.SIGTERM)
            shell.wait(timeout=5)


def test_terminating_pipeline_stops_active_stage_and_prevents_rl(pipeline_env):
    record = Path(pipeline_env["RECORD_ROOT"])
    pipeline_env["SFT_RELEASE"] = str(record / "release")
    with (record / "pipeline.log").open("w") as log:
        process = subprocess.Popen(["bash", str(PIPELINE)], env=pipeline_env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for(lambda: (record / "sft.args").exists())
            process.terminate()
            process.wait(timeout=15)
            assert process.returncode == 143
            assert not (record / "rl.args").exists()
            assert (pipeline_dir(pipeline_env) / "status").read_text().strip() == "stopped"
            stage = pipeline_dir(pipeline_env) / "stages/sft"
            assert (stage / "status").read_text().strip() == "stopped"
        finally:
            Path(pipeline_env["SFT_RELEASE"]).touch()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
