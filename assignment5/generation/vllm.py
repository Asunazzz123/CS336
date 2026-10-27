"""
Small vLLM helpers for server lifecycle, completion requests, and NCCL weight sync.
"""

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from utils.device import get_device_id
from .interface import TokenCompletion as VLLMCompletion

logger = logging.getLogger(__name__)


@dataclass
class VLLMServer:
    model_id: str
    host: str = "127.0.0.1"
    port: int = 8000
    gpu: int = 1
    seed: int = 0
    load_format: str = "auto"
    logging_level: str = "ERROR"
    gpu_memory_utilization: float = 0.6
    max_model_len: int = 2048
    max_num_seqs: int = 32
    enforce_eager: bool = True
    launch_server: bool = True
    startup_timeout: int = 600
    shutdown_timeout: int = 30

    def __post_init__(self) -> None:
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None
        self.weight_sync_group = None
        config_file = Path(self.model_id) / "config.json"
        self._requires_policy_sync = config_file.is_file() and json.loads(
            config_file.read_text()
        ).get("model_type") == "qwen3_5_text"
        self._weights_synchronized = False

    def start(self) -> None:
        if self.launch_server:
            with socket.socket() as probe:
                try:
                    probe.bind((self.host, self.port))
                except OSError as error:
                    raise RuntimeError("vLLM port is occupied; refusing to replace another server") from error
            self.process = start_server(
                model_id=self.model_id,
                host=self.host,
                port=self.port,
                gpu=self.gpu,
                seed=self.seed,
                load_format=self.load_format,
                logging_level=self.logging_level,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                enforce_eager=self.enforce_eager,
            )
            atexit.register(self.stop)
        wait_for_server(self.base_url, self.process, self.startup_timeout)

    def stop(self) -> None:
        stop_server(self.process, timeout=self.shutdown_timeout)
        self.process = None
        group, self.weight_sync_group = self.weight_sync_group, None
        if group is not None and hasattr(group, "destroy"):
            group.destroy()

    def init_weight_sync(self, policy_device: str):
        self.weight_sync_group = init_weight_sync(self.base_url, policy_device)
        return self.weight_sync_group

    def sync_policy_weights(self, policy: torch.nn.Module) -> None:
        if self.weight_sync_group is None:
            raise RuntimeError("Call init_weight_sync before sync_policy_weights.")
        self._weights_synchronized = False
        sync_policy_weights(policy, self.base_url, self.weight_sync_group)
        self._weights_synchronized = True

    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]:
        if self._requires_policy_sync and not self._weights_synchronized:
            raise RuntimeError("Synchronize the text policy before requesting completions from the dummy wrapper")
        return generate_completions(
            vllm_base_url=self.base_url,
            model_id=self.model_id,
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body)


def kill_existing_vllm_server(port: int) -> None:
    pattern = f"vllm serve .* --port {port}"
    try:
        result = subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
        if result.returncode == 0:
            time.sleep(2)
            subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)
    except FileNotFoundError:
        pass


def _qwen35_serving_source(model_id: str) -> tuple[str, bool]:
    """Resolve text checkpoints to vLLM 0.19's registered Qwen3.5 wrapper.

    SFT exports a text-only config. Follow local provenance to the original
    multimodal config, keeping the checkpoint tokenizer and public model name.
    No weights are copied or downloaded. The dummy wrapper must receive a full
    policy sync before the first completion request.
    """
    path = Path(model_id)
    if not (path / "config.json").is_file():
        return model_id, False
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") == "qwen3_5":
        return model_id, True
    if config.get("model_type") != "qwen3_5_text":
        return model_id, False

    explicit = os.environ.get("VLLM_BASE_MODEL_PATH")
    pending = [Path(explicit)] if explicit else [path]
    visited = set()
    dimensions = ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "head_dim",
                  "linear_num_key_heads", "linear_num_value_heads",
                  "linear_key_head_dim", "linear_value_head_dim")
    while pending:
        candidate = pending.pop(0).expanduser().resolve()
        if candidate in visited:
            continue
        visited.add(candidate)
        config_file = candidate / "config.json"
        if not config_file.is_file():
            continue
        source = json.loads(config_file.read_text())
        if source.get("model_type") == "qwen3_5":
            text = source["text_config"]
            if any(config.get(key) != text.get(key) for key in dimensions):
                raise ValueError("Qwen3.5 serving source dimensions differ from the policy checkpoint")
            if config.get("tie_word_embeddings", False) != text.get(
                "tie_word_embeddings", source.get("tie_word_embeddings", False)
            ):
                raise ValueError("Qwen3.5 serving source has a different embedding tying policy")
            return str(candidate), True
        for filename, key in (("text_export_verification.json", "source"),
                              ("config.json.run", "model_id")):
            provenance = candidate / filename
            if provenance.is_file():
                value = json.loads(provenance.read_text()).get(key)
                if isinstance(value, str) and value:
                    target = Path(value).expanduser()
                    if not target.is_absolute() and not target.exists():
                        target = candidate / target
                    pending.append(target)
    raise ValueError(
        "vLLM 0.19 requires the original Qwen3.5 multimodal config for a text policy. "
        "Set VLLM_BASE_MODEL_PATH to the matching local Base directory or retain its provenance."
    )


def start_server(
    model_id: str,
    host: str,
    port: int,
    gpu: int,
    seed: int,
    load_format: str,
    logging_level: str,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    max_num_seqs: int = 32,
    enforce_eager: bool = True,
) -> subprocess.Popen:
    if max_model_len <= 0 or max_num_seqs <= 0:
        raise ValueError("vLLM context length and sequence limit must be positive.")
    model_id = str(model_id)
    serving_source, qwen_wrapper = _qwen35_serving_source(model_id)
    serving_format = "dummy" if serving_source != model_id and qwen_wrapper else load_format
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else None
    if devices is not None and not 0 <= gpu < len(devices):
        raise ValueError("vLLM GPU must be a local ordinal in CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = devices[gpu] if devices else str(gpu)
    # The owned inference server is a separate world, not another FSDP rank.
    for name in list(env):
        if name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE") or name.startswith("TORCHELASTIC_"):
            env.pop(name, None)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_LOGGING_LEVEL"] = logging_level
    project = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        "vllm",
        "serve",
        serving_source,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--enable-prefix-caching",
        # Qwen3.5 supports aligned recurrent-state caching, not the 'all' mode.
        "--mamba-cache-mode",
        "align",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--seed",
        str(seed),
        "--tensor-parallel-size",
        "1",
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),
        "--load-format",
        serving_format,
    ]
    if qwen_wrapper:
        command.extend(["--language-model-only", "--served-model-name", model_id,
                        "--tokenizer", model_id])
        if (Path(model_id) / "generation_config.json").is_file():
            command.extend(["--generation-config", model_id])
    if env.get("CS336_VLLM_WEIGHT_PROBE") == "1":
        command.extend(["--worker-extension-cls", "generation.worker.WeightSyncProbe"])
    if enforce_eager:
        command.append("--enforce-eager")
    logger.info("Starting vLLM server: %s", " ".join(command))
    return subprocess.Popen(command, env=env, start_new_session=True)


def wait_for_server(base_url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}.")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM server at {base_url}.")


def stop_server(process: subprocess.Popen | None, timeout: int = 30) -> None:
    if process is None:
        return
    # Popen owns a new session. Even an exited API leader may have inference
    # workers left in its process group; stop them before handing GPUs back.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def generate_completions(
    vllm_base_url: str,
    model_id: str,
    prompts: list,
    sampling_params: dict,
    batch_size: int | None = None,
) -> list[VLLMCompletion]:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    prompt_batches = [prompts]
    if batch_size is not None:
        prompt_batches = [prompts[start : start + batch_size] for start in range(0, len(prompts), batch_size)]

    completions = []
    for prompt_batch in prompt_batches:
        payload = {
            "model": model_id,
            "prompt": prompt_batch,
            "temperature": sampling_params["temperature"],
            "max_tokens": sampling_params["max_tokens"],
            "n": sampling_params["n"],
            "seed": sampling_params["seed"],
            "return_token_ids": True,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        }
        if sampling_params.get("stop") is not None:
            payload["stop"] = sampling_params["stop"]
            payload["include_stop_str_in_output"] = sampling_params.get("include_stop_str_in_output", False)

        response = _http_json("POST", f"{vllm_base_url}/v1/completions", payload, timeout=3600)
        choices = sorted(response["choices"], key=lambda choice: choice["index"])
        completions.extend(
            VLLMCompletion(
                text=choice["text"],
                token_ids=choice.get("token_ids") or [],
                finish_reason=choice.get("finish_reason"),
            )
            for choice in choices
        )
    return completions


def init_weight_sync(vllm_base_url: str, policy_device: str):
    device_id = get_device_id(policy_device)
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port

    inference_world_size = _http_json("GET", f"{vllm_base_url}/get_world_size", timeout=10)["world_size"]
    world_size = inference_world_size + 1
    master_address = get_ip()
    master_port = get_open_port()
    init_info = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": 1,
        "world_size": world_size,
    }

    torch.cuda.set_device(device_id)
    with ThreadPoolExecutor(max_workers=1) as executor:
        init_future = executor.submit(
            _http_json,
            "POST",
            f"{vllm_base_url}/init_weight_transfer_engine",
            {"init_info": init_info},
            60,
        )
        weight_sync_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        init_future.result()

    return weight_sync_group


def sync_policy_weights(policy: torch.nn.Module, vllm_base_url: str, weight_sync_group) -> None:
    device_id = get_device_id(next(policy.parameters()).device)
    """Copy policy weights into vLLM and invalidate caches derived from old weights."""
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerSendWeightsArgs,
        NCCLWeightTransferEngine,
    )

    # Keep references to the parameters; convert one weight at a time on the
    # sender instead of retaining an entire BF16 copy of the FP32 policy.
    qwen_text = getattr(getattr(policy, "config", None), "model_type", None) == "qwen3_5_text"
    weights = [
        ("model.language_model." + name[len("model."):]
         if qwen_text and name.startswith("model.") else name, tensor)
        for name, tensor in policy.named_parameters()
    ]
    transfer_dtypes = [torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
                       for _, tensor in weights]
    update_info = {
        "names": [name for name, _ in weights],
        "dtype_names": [str(dtype).split(".")[-1] for dtype in transfer_dtypes],
        "shapes": [list(tensor.shape) for _, tensor in weights],
        "packed": True,
        "is_checkpoint_format": True,
    }

    inference_weights = (
        (name, tensor.detach().to(dtype=dtype).contiguous())
        for (name, tensor), dtype in zip(weights, transfer_dtypes)
    )

    torch.cuda.set_device(device_id)
    _http_json("POST", f"{vllm_base_url}/pause", timeout=60)
    sync_error = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            update_future = executor.submit(
                _http_json,
                "POST",
                f"{vllm_base_url}/update_weights",
                {"update_info": update_info},
                300,
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=inference_weights,
                trainer_args=NCCLTrainerSendWeightsArgs(
                    group=weight_sync_group,
                    packed=True,
                ),
            )
            update_future.result()
    except BaseException as error:
        sync_error = error
        raise
    finally:
        # A failed transfer may still have replaced some weights. Invalidate
        # derived state and always attempt to release the paused server. The
        # caller receives the transfer error and must not continue training.
        cleanup_error = None
        for endpoint in ("reset_prefix_cache", "resume"):
            try:
                _http_json("POST", f"{vllm_base_url}/{endpoint}", timeout=60)
            except Exception as error:
                logger.exception("vLLM cleanup failed at %s", endpoint)
                cleanup_error = cleanup_error or error
        if sync_error is None and cleanup_error is not None:
            raise cleanup_error


def sync_checkpoint_weights(path, vllm_base_url, weight_sync_group, policy_device):
    """Stream an HF CPU snapshot into a TP receiver without a training model.

    This uses the pinned vLLM 0.19.1 transfer API. Packed=False bounds staging
    to one tensor on both sender/receiver and avoids packed-loader tied aliases.
    """
    from safetensors import safe_open
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    index = path / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as file:
            mapping = {name: "model.safetensors" for name in file.keys()}
    if config.get("tie_word_embeddings") and any("embed_tokens.weight" in name or name == "transformer.wte.weight" for name in mapping):
        mapping.pop("lm_head.weight", None)
    def serving_name(name):
        if config.get("model_type") == "qwen3_5_text" and name.startswith("model."):
            return "model.language_model." + name[len("model."):]
        return name
    names, shapes, dtypes = [], [], []
    # Read headers only when constructing transfer metadata.
    for name, filename in mapping.items():
        with safe_open(str(path / filename), framework="pt", device="cpu") as file:
            view = file.get_slice(name)
            names.append(serving_name(name))
            shapes.append(view.get_shape())
            dtypes.append({"BF16": "bfloat16", "F32": "float32", "F16": "float16",
                           "I64": "int64", "I32": "int32", "BOOL": "bool"}[view.get_dtype()])
    def tensors():
        for name, filename in mapping.items():
            with safe_open(str(path / filename), framework="pt", device="cpu") as file:
                yield serving_name(name), file.get_tensor(name).to(policy_device).contiguous()
    torch.cuda.set_device(get_device_id(policy_device))
    _http_json("POST", f"{vllm_base_url}/pause")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_http_json, "POST", f"{vllm_base_url}/update_weights", {
                "update_info": {"names": names, "shapes": shapes, "dtype_names": dtypes,
                                "packed": False, "is_checkpoint_format": True}}, 300)
            NCCLWeightTransferEngine.trainer_send_weights(tensors(), NCCLTrainerSendWeightsArgs(
                group=weight_sync_group, packed=False))
            future.result()
        _http_json("POST", f"{vllm_base_url}/reset_prefix_cache")
        _http_json("POST", f"{vllm_base_url}/resume")
    except BaseException:
        # The owning backend shuts down an uncertain server; never serve it.
        raise


class VLLMBackend:
    """Owned, versioned server. No imports from distributed or alignment.

    Each load starts from the supplied snapshot. close releases all inference
    GPU allocations before the training shards return to CUDA.
    """
    def __init__(self, *, host="127.0.0.1", port=8000, gpu=1, policy_device="cuda:0",
                 gpu_memory_utilization=0.8, max_model_len=2048, max_num_seqs=32,
                 startup_timeout=600):
        self.options = dict(host=host, port=port, gpu=gpu,
            gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
            max_num_seqs=max_num_seqs, startup_timeout=startup_timeout)
        self.policy_device = policy_device
        self.server, self.version = None, None

    def load_policy(self, snapshot):
        if self.server is not None:
            raise RuntimeError("Close the previous generation session first")
        version = json.loads((snapshot.path / "policy_version.json").read_text())["policy_version"]
        if version != snapshot.version:
            raise ValueError("Snapshot policy version does not match request")
        self.server = VLLMServer(str(snapshot.path), **self.options)
        try:
            self.server.start()
            if self.server._requires_policy_sync:
                self.server.init_weight_sync(self.policy_device)
                sync_checkpoint_weights(snapshot.path, self.server.base_url,
                                        self.server.weight_sync_group, self.policy_device)
                self.server._weights_synchronized = True
            self.version = version
            return version
        except BaseException:
            self.close()
            raise

    def generate(self, prompt_ids, sampling):
        if self.server is None or self.version is None:
            raise RuntimeError("Load and acknowledge a policy before generation")
        responses = self.server.generate_completions(prompt_ids, sampling)
        if len(responses) != len(prompt_ids) or any(not r.token_ids for r in responses):
            raise ValueError("vLLM must return one nonempty token-ID response per prompt")
        return responses

    def close(self):
        self.version = None
        if self.server is not None:
            server, self.server = self.server, None
            server.stop()
