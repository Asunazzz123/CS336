"""Single-device training runtime; inference implementation belongs to generation."""
import socket
import torch
from training.model_factory import load_model_and_tokenizer
from generation import vllm as vllm_utils
from training.types import Runtime
from training.configs import RuntimeConfig as TrainConfig
from training.optim.factory import optimizer_switcher


def build_runtime(config: TrainConfig) -> Runtime:
    # 1. 先将训练模型放到目标设备，再创建优化器
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA")
    model, tokenizer = load_model_and_tokenizer(config.model_id, dtype=getattr(torch, config.dtype), device=device)
    if config.gradient_checkpointing:
        # 只在 train 模式下生效；train_on_rollouts 在开启检查点时会切到 train 模式
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = optimizer_switcher(model, config)
    # 2. Transformers 复用模型；vLLM 在另一块 CUDA GPU 上生成
    if config.generation_backend == "transformers":
        return Runtime(model, tokenizer, optimizer, model)
    if config.generation_backend != "vllm":
        raise ValueError("未知生成后端")
    if device.type != "cuda":
        raise ValueError("项目的 vLLM NCCL 同步路径要求 CUDA 训练设备")
    policy_gpu = device.index if device.index is not None else torch.cuda.current_device()
    if policy_gpu == config.vllm_gpu:
        raise ValueError("训练与 vLLM 需要不同的 GPU")
    # 不调用会终止已有服务的 VLLMServer.start，仅启动本次拥有的进程
    server = vllm_utils.VLLMServer(
        config.model_id, host=config.vllm_host, port=config.vllm_port,
        gpu=config.vllm_gpu, seed=config.seed,
    )
    # 已占用端口直接报错，不终止或接管其他进程
    with socket.socket() as probe:
        try:
            probe.bind((config.vllm_host, config.vllm_port))
        except OSError as error:
            raise ValueError("vLLM 端口不可用，请选择另一个 --vllm-port") from error
    try:
        server.process = vllm_utils.start_server(
            config.model_id, config.vllm_host, config.vllm_port,
            config.vllm_gpu, config.seed, "auto", "ERROR",
            gpu_memory_utilization=server.gpu_memory_utilization,
            max_model_len=server.max_model_len,
            max_num_seqs=server.max_num_seqs,
            enforce_eager=server.enforce_eager,
        )
        vllm_utils.wait_for_server(server.base_url, server.process, server.startup_timeout)
        server.init_weight_sync(str(device))
    except BaseException:
        server.stop()
        raise
    return Runtime(model, tokenizer, optimizer, server)


def close_runtime(runtime):
    if isinstance(runtime.generator, vllm_utils.VLLMServer):
        runtime.generator.stop()
