"""Version/phase boundaries for FSDP -> independent generation receivers."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from distributed.generation import ExternalGeneration
from distributed import Phase
from generation import PolicySnapshot
from generation import vllm
from checkpointing.best import validate_hf_checkpoint
from .test_distributed_runtime import local_engine
from .test_fsdp_grpo_integration import tiny_model
from .test_vllm_utils import install_transfer_stub
from utils.device import nvidia_cuda_available


@pytest.mark.parametrize("failure", [None, "stale", "generate", "close"])
def test_external_generation_parks_before_load_and_closes_before_resume(tmp_path, failure):
    engine = local_engine(tiny_model())
    engine.policy_version = 7
    engine.context.gather_objects = lambda value: [value]
    events = []
    def park():
        events.append("park")
        engine.phase = Phase.PARKED
    def resume():
        assert events[-1] == "close"
        events.append("resume")
        engine.phase = Phase.READY
    engine.park, engine.resume = park, resume
    class Backend:
        def load_policy(self, snapshot):
            assert engine.phase == Phase.PARKED
            validate_hf_checkpoint(snapshot.path)
            assert snapshot.version == 7
            events.append("load")
            return 6 if failure == "stale" else 7
        def generate(self, ids, sampling):
            events.append("generate")
            assert ids == [[2, 3], [4]]
            if failure == "generate":
                raise RuntimeError("generate")
            return [vllm.VLLMCompletion("a", [5, 1], "stop"),
                    vllm.VLLMCompletion("b", [6], "length")]
        def close(self):
            events.append("close")
            if failure == "close":
                raise RuntimeError("close")
    adapter = ExternalGeneration(engine, SimpleNamespace(pad_token_id=0), Backend,
        tmp_path / "snapshots", source_model="original-base")
    # save_assets is supplied by the real tiny model; tokenizer is unnecessary here.
    adapter.tokenizer = None
    export = engine.export_model_only_hf
    def capture(path, **kwargs):
        return export(path)
    engine.export_model_only_hf = capture
    adapter.tokenizer = SimpleNamespace(pad_token_id=0)
    if failure:
        with pytest.raises(RuntimeError):
            adapter.generate([[2, 3], [4]], temperature=1., max_new_tokens=3, seed=4)
        assert engine.phase == Phase.FAILED
        assert "resume" not in events
    else:
        result = adapter.generate([[2, 3], [4]], temperature=1., max_new_tokens=3, seed=4)
        assert events == ["park", "load", "generate", "close", "resume"]
        assert result.policy_version == 7 and result.rollout_id == 0
        assert result.response_ids.tolist() == [[5, 1], [6, 0]]
        assert result.response_mask.tolist() == [[True, True], [True, False]]
        assert engine.rollout_id == 1
    assert not list((tmp_path / "snapshots").glob("policy-*"))


def test_vllm_isolates_inference_world_and_maps_visible_gpu_ordinals(monkeypatch):
    for name, value in {"CUDA_VISIBLE_DEVICES": "2,3", "RANK": "0", "WORLD_SIZE": "2",
                         "LOCAL_RANK": "0", "MASTER_PORT": "29500", "TORCHELASTIC_RUN_ID": "job"}.items():
        monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(vllm.subprocess, "Popen", lambda command, **kwargs: calls.append(kwargs))
    vllm.start_server("regular-model", "127.0.0.1", 8000, 1, 1, "auto", "ERROR")
    env = calls[0]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert not any(name in env for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_PORT", "TORCHELASTIC_RUN_ID"))


def test_checkpoint_nccl_transfer_streams_qwen_names_and_skips_tied_head(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    path = tmp_path / "snapshot"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text", "tie_word_embeddings": True}))
    weights = {"model.embed_tokens.weight": torch.randn(4, 3).bfloat16(),
               "lm_head.weight": torch.randn(4, 3).bfloat16(),
               "model.layers.0.projection.weight": torch.randn(3, 3).bfloat16()}
    save_file(weights, str(path / "model.safetensors"))
    requests, names = [], []
    def send(iterator, trainer_args):
        assert trainer_args.packed is False
        for name, tensor in iterator:
            names.append(name)
            assert tensor.device.type == "cpu" and tensor.dtype == torch.bfloat16
    install_transfer_stub(monkeypatch, send)
    monkeypatch.setattr(vllm, "_http_json", lambda method, url, payload=None, timeout=60: requests.append((url, payload)))
    vllm.sync_checkpoint_weights(path, "http://unused", "group", "cpu")
    assert names == ["model.language_model.embed_tokens.weight", "model.language_model.layers.0.projection.weight"]
    info = next(payload["update_info"] for url, payload in requests if url.endswith("update_weights"))
    assert info["names"] == names and info["packed"] is False
    assert [url.rsplit("/", 1)[-1] for url, _ in requests] == ["pause", "update_weights", "reset_prefix_cache", "resume"]


def test_versioned_vllm_backend_requires_full_load_and_stops_owned_server(tmp_path, monkeypatch):
    (tmp_path / "policy_version.json").write_text(json.dumps({"policy_version": 3}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "regular"}))
    events = []
    monkeypatch.setattr(vllm.VLLMServer, "start", lambda self: events.append("start"))
    monkeypatch.setattr(vllm.VLLMServer, "stop", lambda self: events.append("stop"))
    monkeypatch.setattr(vllm.VLLMServer, "generate_completions", lambda *args: [vllm.VLLMCompletion("a", [5, 1], "stop")])
    backend = vllm.VLLMBackend()
    with pytest.raises(ValueError, match="version"):
        backend.load_policy(PolicySnapshot(tmp_path, 2))
    assert backend.load_policy(PolicySnapshot(tmp_path, 3)) == 3
    assert backend.generate([[1, 2]], {})[0].token_ids == [5, 1]
    backend.close()
    assert events == ["start", "stop"]
    with pytest.raises(RuntimeError, match="Load"):
        backend.generate([[1]], {})


def test_server_stop_reaps_workers_even_when_api_leader_has_exited(monkeypatch):
    import signal
    signals, waits = [], []
    process = SimpleNamespace(pid=12345, poll=lambda: 0, wait=lambda **kwargs: waits.append(kwargs))
    monkeypatch.setattr(vllm.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    vllm.stop_server(process, timeout=3)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert waits == [{"timeout": 3}]


@pytest.mark.skipif(not nvidia_cuda_available() or torch.cuda.device_count() < 2
                    or importlib.util.find_spec("vllm") is None,
                    reason="Requires two NVIDIA GPUs and pinned vLLM")
def test_real_two_gpu_sft_vllm_grpo_pipeline(tmp_path):
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=2", "-m", "tests.fsdp_pipeline_worker", "--output-dir", str(tmp_path / "run")],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=1800, env=dict(os.environ, OMP_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-12000:]
    assert json.loads((tmp_path / "run/passed.json").read_text())["grpo_policy_version"] == 2
