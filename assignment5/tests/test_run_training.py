"""Exercise the shell launcher without downloading models or running Python training."""

import os
from pathlib import Path
import signal
import subprocess
import time

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "run_training.sh"


def _wait_for(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("launcher did not reach the expected state before timeout")


@pytest.fixture
def launcher_env(tmp_path):
    fake_conda = tmp_path / "fake conda"
    fake_conda.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$PWD" "$@" > "$FAKE_RECORD"
printf 'fake training started\\n'
if [[ -n "${FAKE_GRANDCHILD_PID:-}" ]]; then
    bash -c 'trap "" TERM; echo "$$" > "$FAKE_GRANDCHILD_PID"; while [[ ! -f "$FAKE_RELEASE" ]]; do sleep 0.05; done' &
fi
if [[ -n "${FAKE_RELEASE:-}" ]]; then
    while [[ ! -f "$FAKE_RELEASE" ]]; do sleep 0.05; done
fi
printf 'fake training finished\\n'
exit "${FAKE_EXIT:-0}"
""",
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    env = {
        # The fake subprocess needs no credentials or unrelated host settings.
        "PATH": os.environ["PATH"],
        "CONDA_EXE": str(fake_conda),
        "CONDA_ENV": "agent",
        "RUN_ROOT": str(tmp_path / "training jobs"),
        "RUN_NAME": "test-job",
        "FAKE_RECORD": str(tmp_path / "received arguments"),
    }
    yield env
    if "FAKE_RELEASE" in env:
        Path(env["FAKE_RELEASE"]).touch()
    run_root = Path(env["RUN_ROOT"])
    if run_root.exists():
        for job_dir in run_root.iterdir():
            if job_dir.is_dir() and (job_dir / "pid").exists():
                subprocess.run(
                    ["bash", str(LAUNCHER), "stop", str(job_dir)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )


def _launch(env, *args, cwd=None):
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        env=env,
        cwd=cwd or "/tmp",
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _job_dir(env):
    return Path(env["RUN_ROOT"]) / env["RUN_NAME"]


def _wait_for_exit(env, expected):
    exit_file = _job_dir(env) / "exit_code"
    _wait_for(lambda: exit_file.exists() and exit_file.read_text().strip() == str(expected))


@pytest.mark.parametrize(
    ("mode", "entry"),
    [("sft", "train_sft.py"), ("grpo", "train_grpo.py"), ("grpo-full", "train_grpo_full.py")],
)
def test_run_selects_entry_and_preserves_argument_boundaries(launcher_env, tmp_path, mode, entry):
    model = "SFT checkpoint with spaces;$HOME"
    output_dir = tmp_path / "model checkpoints"
    result = _launch(
        launcher_env, "run", mode,
        "--model-id", model, "--output-dir", str(output_dir),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    received = Path(launcher_env["FAKE_RECORD"]).read_text().splitlines()
    assert received[0] == str(PROJECT_ROOT)
    args = received[1:]
    assert args[0] == "run"
    assert args[args.index("-n") + 1] == "agent"
    assert "python" in args
    assert any(Path(arg).name == entry for arg in args)
    assert args[args.index("--model-id") + 1] == model
    assert args[args.index("--output-dir") + 1] == str(output_dir)
    _wait_for_exit(launcher_env, 0)


def test_default_entry_and_artifacts_are_isolated_from_control_files(launcher_env):
    result = _launch(launcher_env)
    assert result.returncode == 0, result.stdout + result.stderr
    _wait_for_exit(launcher_env, 0)
    received = Path(launcher_env["FAKE_RECORD"]).read_text().splitlines()
    assert any(Path(arg).name == "train_grpo_full.py" for arg in received)
    assert Path(received[received.index("--output-dir") + 1]) == _job_dir(launcher_env) / "artifacts"
    assert "fake training finished" in (_job_dir(launcher_env) / "train.log").read_text()


def test_training_failure_is_saved_and_foreground_returns_same_code(launcher_env):
    launcher_env["FAKE_EXIT"] = "23"
    result = _launch(launcher_env, "run", "sft")
    assert result.returncode == 23, result.stdout + result.stderr
    _wait_for_exit(launcher_env, 23)
    status = _launch(launcher_env, "status")
    assert "failed" in (status.stdout + status.stderr).lower()
    assert "23" in status.stdout + status.stderr


def test_background_failure_remains_observable_after_start_returns(launcher_env):
    launcher_env["FAKE_EXIT"] = "19"
    result = _launch(launcher_env, "start", "grpo")
    assert result.returncode == 0, result.stdout + result.stderr
    _wait_for_exit(launcher_env, 19)
    status = _launch(launcher_env, "status", launcher_env["RUN_NAME"])
    assert "failed" in (status.stdout + status.stderr).lower()
    assert "19" in status.stdout + status.stderr


def test_background_training_survives_hangup_of_launching_shell(launcher_env, tmp_path):
    release = tmp_path / "release"
    ready = tmp_path / "shell ready"
    launcher_env["FAKE_RELEASE"] = str(release)
    # A separate session lets this test emulate the launching SSH process group
    # receiving SIGHUP without delivering it to pytest itself.
    shell = subprocess.Popen(
        [
            "bash", "-c",
            'bash "$1" start sft; touch "$2"; while :; do sleep 0.1; done',
            "launching-shell", str(LAUNCHER), str(ready),
        ],
        env=launcher_env,
        cwd="/tmp",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_for(lambda: ready.exists() and Path(launcher_env["FAKE_RECORD"]).exists())
        os.killpg(shell.pid, signal.SIGHUP)
        shell.wait(timeout=5)
        assert shell.returncode != 0
        time.sleep(0.15)
        assert not (_job_dir(launcher_env) / "exit_code").exists()
        release.touch()
        _wait_for_exit(launcher_env, 0)
        assert "fake training finished" in (_job_dir(launcher_env) / "train.log").read_text()
    finally:
        release.touch()
        if shell.poll() is None:
            os.killpg(shell.pid, signal.SIGTERM)
            shell.wait(timeout=5)


def test_status_logs_stop_and_duplicate_job_name(launcher_env, tmp_path):
    release = tmp_path / "release"
    launcher_env["FAKE_RELEASE"] = str(release)
    result = _launch(launcher_env, "start", "grpo")
    assert result.returncode == 0, result.stdout + result.stderr
    _wait_for(lambda: Path(launcher_env["FAKE_RECORD"]).exists())
    for reference in ((), (launcher_env["RUN_NAME"],), (str(_job_dir(launcher_env)),)):
        status = _launch(launcher_env, "status", *reference)
        assert status.returncode == 0, status.stdout + status.stderr
        assert "Status: running" in status.stdout.splitlines()
    logs = _launch(launcher_env, "logs", launcher_env["RUN_NAME"])
    assert logs.returncode == 0, logs.stdout + logs.stderr
    assert "fake training started" in logs.stdout
    duplicate = _launch(launcher_env, "start", "sft")
    assert duplicate.returncode != 0
    assert "fake training started" in (_job_dir(launcher_env) / "train.log").read_text()
    stopped = _launch(launcher_env, "stop", launcher_env["RUN_NAME"])
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    _wait_for(lambda: (_job_dir(launcher_env) / "exit_code").exists())
    assert int((_job_dir(launcher_env) / "exit_code").read_text().strip()) != 0
    status = _launch(launcher_env, "status")
    assert "stopped" in (status.stdout + status.stderr).lower()


def test_invalid_command_does_not_start_training(launcher_env):
    result = _launch(launcher_env, "unknown-command")
    assert result.returncode != 0
    assert not Path(launcher_env["FAKE_RECORD"]).exists()


def test_stop_cleans_up_a_descendant_that_ignores_term(launcher_env, tmp_path):
    launcher_env["FAKE_RELEASE"] = str(tmp_path / "release")
    child_pid_file = tmp_path / "grandchild pid"
    launcher_env["FAKE_GRANDCHILD_PID"] = str(child_pid_file)
    started = _launch(launcher_env, "start", "grpo")
    assert started.returncode == 0, started.stdout + started.stderr
    _wait_for(lambda: child_pid_file.exists() and child_pid_file.read_text().strip())
    child_pid = int(child_pid_file.read_text())
    stopped = _launch(launcher_env, "stop")
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr

    def child_has_exited():
        state = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "stat="],
            capture_output=True, text=True, check=False,
        )
        return not state.stdout.strip() or state.stdout.lstrip().startswith("Z")

    _wait_for(child_has_exited)


def test_stop_refuses_a_pid_that_belongs_to_an_unrelated_process(launcher_env, tmp_path):
    launcher_env["FAKE_RELEASE"] = str(tmp_path / "release")
    started = _launch(launcher_env, "start", "sft")
    assert started.returncode == 0, started.stdout + started.stderr
    _wait_for(lambda: Path(launcher_env["FAKE_RECORD"]).exists())
    pid_file = _job_dir(launcher_env) / "pid"
    original_pid = pid_file.read_text()
    unrelated = subprocess.Popen(
        ["bash", "-c", "while :; do sleep 0.05; done"],
        env=launcher_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        pid_file.write_text(f"{unrelated.pid}\n")
        stopped = _launch(launcher_env, "stop")
        assert stopped.returncode != 0
        assert unrelated.poll() is None
    finally:
        pid_file.write_text(original_pid)
        if unrelated.poll() is None:
            os.killpg(unrelated.pid, signal.SIGTERM)
        unrelated.wait(timeout=5)
