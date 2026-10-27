"""Exercise launch and weight-transfer contracts without vLLM or CUDA."""

import sys
import json
from types import GeneratorType, ModuleType, SimpleNamespace

import pytest
import torch

from alignment import vllm_utils


def install_transfer_stub(monkeypatch, send):
    path = "vllm.distributed.weight_transfer.nccl_engine"
    module = ModuleType(path)
    module.NCCLTrainerSendWeightsArgs = SimpleNamespace
    module.NCCLWeightTransferEngine = SimpleNamespace(trainer_send_weights=send)
    monkeypatch.setitem(sys.modules, path, module)
    # CPU tensors stand in for policy weights; fake the CUDA boundary explicitly.
    monkeypatch.setattr(vllm_utils, "get_device_id", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)


def test_weight_sync_rejects_cpu_before_importing_backend_or_calling_http(monkeypatch):
    monkeypatch.setattr(vllm_utils, "_http_json", lambda *a, **k: pytest.fail("Unexpected HTTP"))
    with pytest.raises(ValueError, match="CUDA device"):
        vllm_utils.init_weight_sync("http://unused", "cpu")
    with pytest.raises(ValueError, match="CUDA device"):
        vllm_utils.sync_policy_weights(torch.nn.Linear(2, 2), "http://unused", "unused")


def test_server_limits_qwen_recurrent_cache_and_eager_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_utils.subprocess, "Popen", lambda command, **kwargs: calls.append((command, kwargs)))
    vllm_utils.start_server("text-checkpoint", "127.0.0.1", 8001, 1, 7, "auto", "ERROR")
    command, kwargs = calls[0]
    assert command[command.index("--max-model-len") + 1] == "2048"
    assert command[command.index("--max-num-seqs") + 1] == "32"
    assert command[command.index("--mamba-cache-mode") + 1] == "align"
    assert "--enforce-eager" in command
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert kwargs["env"]["VLLM_SERVER_DEV_MODE"] == "1"
    assert kwargs["start_new_session"] is True


def test_server_allows_explicit_context_and_graph_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_utils.subprocess, "Popen", lambda command, **kwargs: calls.append(command))
    vllm_utils.start_server("text-checkpoint", "127.0.0.1", 8001, 1, 7, "auto", "ERROR",
                            max_model_len=4096, max_num_seqs=8, enforce_eager=False)
    assert calls[0][calls[0].index("--max-model-len") + 1] == "4096"
    assert calls[0][calls[0].index("--max-num-seqs") + 1] == "8"
    assert "--enforce-eager" not in calls[0]


def test_sync_streams_bf16_weights_and_preserves_fp32_tied_policy(monkeypatch):
    policy = torch.nn.Module()
    policy.model = torch.nn.Embedding(4, 3)
    policy.lm_head = torch.nn.Linear(3, 4, bias=False)
    policy.lm_head.weight = policy.model.weight
    policy.projection = torch.nn.Linear(3, 2)
    original = {name: value.detach().clone() for name, value in policy.named_parameters()}
    requests, received = [], []

    def http(method, url, payload=None, timeout=60):
        requests.append((url.rsplit("/", 1)[-1], payload))
        return {}

    def send(iterator, trainer_args):
        assert isinstance(iterator, GeneratorType)
        assert trainer_args.group == "group"
        assert trainer_args.packed
        for name, tensor in iterator:
            assert tensor.dtype == torch.bfloat16
            assert not tensor.requires_grad
            assert tensor.is_contiguous()
            assert torch.equal(tensor, original[name].to(torch.bfloat16))
            received.append(name)

    monkeypatch.setattr(vllm_utils, "_http_json", http)
    install_transfer_stub(monkeypatch, send)
    vllm_utils.sync_policy_weights(policy, "http://vllm", "group")
    assert received == list(original)
    assert "lm_head.weight" not in received  # The server shares the same embedding.
    info = next(payload["update_info"] for endpoint, payload in requests if endpoint == "update_weights")
    assert info["names"] == received
    assert info["dtype_names"] == ["bfloat16"] * len(received)
    assert info["shapes"] == [list(original[name].shape) for name in received]
    assert info["is_checkpoint_format"] is True
    assert [endpoint for endpoint, _ in requests][0] == "pause"
    assert [endpoint for endpoint, _ in requests][-2:] == ["reset_prefix_cache", "resume"]
    for name, parameter in policy.named_parameters():
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter.detach(), original[name])


@pytest.mark.parametrize("failure", ["send", "update_weights", "reset_prefix_cache", "resume"])
def test_failed_sync_still_invalidates_cache_and_attempts_resume(monkeypatch, failure):
    policy = torch.nn.Linear(2, 2)
    requests = []

    def http(method, url, payload=None, timeout=60):
        endpoint = url.rsplit("/", 1)[-1]
        requests.append(endpoint)
        if endpoint == failure:
            raise RuntimeError(failure)
        return {}

    def send(iterator, trainer_args):
        if failure == "send":
            raise RuntimeError("send")
        list(iterator)

    monkeypatch.setattr(vllm_utils, "_http_json", http)
    install_transfer_stub(monkeypatch, send)
    with pytest.raises(RuntimeError, match=failure):
        vllm_utils.sync_policy_weights(policy, "http://vllm", "group")
    assert requests[-2:] == ["reset_prefix_cache", "resume"]


def test_cleanup_failure_does_not_hide_transfer_failure(monkeypatch):
    def http(method, url, payload=None, timeout=60):
        if url.endswith("resume"):
            raise RuntimeError("cleanup")
        return {}

    def send(iterator, trainer_args):
        raise ValueError("original transfer failure")

    monkeypatch.setattr(vllm_utils, "_http_json", http)
    install_transfer_stub(monkeypatch, send)
    with pytest.raises(ValueError, match="original transfer failure"):
        vllm_utils.sync_policy_weights(torch.nn.Linear(2, 2), "http://vllm", "group")


def qwen_paths(tmp_path):
    base, text = tmp_path / "base", tmp_path / "text"
    base.mkdir(); text.mkdir()
    dimensions = {"hidden_size": 8, "vocab_size": 16}
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": dimensions}))
    (text / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text", **dimensions}))
    (text / "text_export_verification.json").write_text(json.dumps({"source": str(base)}))
    return base, text


def test_text_checkpoint_uses_registered_wrapper_and_keeps_api_identity(tmp_path, monkeypatch):
    base, text = qwen_paths(tmp_path)
    monkeypatch.delenv("VLLM_BASE_MODEL_PATH", raising=False)
    monkeypatch.setenv("CS336_VLLM_WEIGHT_PROBE", "1")
    commands = []
    monkeypatch.setattr(vllm_utils.subprocess, "Popen", lambda command, **kwargs: commands.append((command, kwargs)))
    vllm_utils.start_server(str(text), "127.0.0.1", 8027, 1, 42, "auto", "ERROR")
    command, kwargs = commands[0]
    assert command[2] == str(base)
    assert command[command.index("--load-format") + 1] == "dummy"
    assert command[command.index("--served-model-name") + 1] == str(text)
    assert command[command.index("--tokenizer") + 1] == str(text)
    assert "--language-model-only" in command
    assert "--worker-extension-cls" in command
    assert kwargs["env"]["PYTHONPATH"].split(":")[0].endswith("assignment5")


def test_sft_checkpoint_follows_local_source_chain(tmp_path, monkeypatch):
    base, text = qwen_paths(tmp_path)
    checkpoint = tmp_path / "sft"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes((text / "config.json").read_bytes())
    (checkpoint / "config.json.run").write_text(json.dumps({"model_id": str(text)}))
    monkeypatch.delenv("VLLM_BASE_MODEL_PATH", raising=False)
    assert vllm_utils._qwen35_serving_source(str(checkpoint)) == (str(base), True)


def test_qwen_source_validation_fails_without_provenance_or_for_wrong_size(tmp_path, monkeypatch):
    base, text = qwen_paths(tmp_path)
    monkeypatch.delenv("VLLM_BASE_MODEL_PATH", raising=False)
    (text / "text_export_verification.json").unlink()
    with pytest.raises(ValueError, match="VLLM_BASE_MODEL_PATH"):
        vllm_utils._qwen35_serving_source(str(text))
    monkeypatch.setenv("VLLM_BASE_MODEL_PATH", str(base))
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": 4}}))
    with pytest.raises(ValueError, match="dimensions differ"):
        vllm_utils._qwen35_serving_source(str(text))


def test_qwen_sync_maps_both_metadata_and_transmitted_names(monkeypatch):
    policy = torch.nn.Module()
    policy.config = SimpleNamespace(model_type="qwen3_5_text")
    policy.model = torch.nn.Module()
    policy.model.embed_tokens = torch.nn.Embedding(4, 3)
    policy.lm_head = torch.nn.Linear(3, 4, bias=False)
    expected = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    sent, requests = [], []
    monkeypatch.setattr(vllm_utils, "_http_json", lambda method, url, payload=None, timeout=60: requests.append((url, payload)))
    install_transfer_stub(monkeypatch, lambda iterator, trainer_args: sent.extend(name for name, tensor in iterator))
    vllm_utils.sync_policy_weights(policy, "http://vllm", "group")
    assert sent == expected
    info = next(payload["update_info"] for url, payload in requests if url.endswith("update_weights"))
    assert info["names"] == expected


def test_dummy_text_wrapper_requires_successful_sync_before_generation(tmp_path, monkeypatch):
    _, text = qwen_paths(tmp_path)
    server = vllm_utils.VLLMServer(str(text))
    server.weight_sync_group = "group"
    monkeypatch.setattr(vllm_utils, "generate_completions", lambda **kwargs: ["ready"])
    monkeypatch.setattr(vllm_utils, "sync_policy_weights", lambda *args: None)
    with pytest.raises(RuntimeError, match="Synchronize the text policy"):
        server.generate_completions(["test"], {})
    server.sync_policy_weights(torch.nn.Linear(2, 2))
    assert server.generate_completions(["test"], {}) == ["ready"]


def test_worker_probe_reads_small_samples_without_modifying_weights():
    from alignment.vllm_worker import WeightSyncProbe
    model = torch.nn.Linear(3, 2, bias=False)
    before = model.weight.detach().clone()
    worker = WeightSyncProbe()
    worker.model_runner = SimpleNamespace(get_model=lambda: model)
    result = worker.weight_sync_probe(["weight"])
    assert result["weight"]["values"] == before.reshape(-1)[:4].tolist()
    assert torch.equal(before, model.weight)
