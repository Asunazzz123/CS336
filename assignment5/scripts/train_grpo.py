"""基础 on-policy GRPO：Transformers / vLLM 生成、训练、验证及 checkpoint。"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import socket
import time
from decimal import Decimal, InvalidOperation
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parents[1]
# 支持直接执行 scripts/train_grpo.py，无需安装当前项目
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from alignment import vllm_utils
from alignment.grpo import run_grpo_train_step


@dataclass
class TrainConfig:
    model_id: str
    train_path: Path
    eval_path: Path
    prompt_path: Path
    output_dir: Path
    device: str = "cpu"
    generation_backend: str = "transformers"
    seed: int = 42
    num_steps: int = 2
    prompts_per_batch: int = 2
    group_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    max_new_tokens: int = 128
    eval_every: int = 1
    eval_examples: int = 4
    save_every: int = 1
    optimizer_name: str = "adamw"
    weight_decay: float = 0.0
    dtype: str = "float32"
    sampling_strategy: str = "random"
    reward_type: str = "gsm8k_numeric"
    generation_batch_size: int = 2
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_gpu: int = 1


@dataclass
class Question:
    question: str
    ground_truth: str


@dataclass
class RolloutBatch:
    # 同题回答连续排列，三个列表一一对应
    repeated_prompts: list[str]
    responses: list[str]
    repeated_ground_truths: list[str]


@dataclass
class Runtime:
    model: Any
    tokenizer: Any
    optimizer: Any
    generator: Any


class CombinedOptimizer:
    """Muon 矩阵参数与 AdamW 辅助参数共享 step / zero_grad 接口。"""
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for optimizer in optimizers for g in optimizer.param_groups]

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state):
        if len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("优化器数量与 checkpoint 不一致")
        for optimizer, saved in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)


def optimizer_switcher(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    kwargs = dict(lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer_name == "adam":
        return torch.optim.Adam(params, **kwargs)
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if config.optimizer_name != "muon":
        raise ValueError(f"未知优化器：{config.optimizer_name}")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("当前 PyTorch 不提供 torch.optim.Muon；请使用 adam/adamw")
    # 排除 embedding / 输出层，Muon 仅更新隐藏层二维矩阵
    excluded = {id(p) for module in (model.get_input_embeddings(), model.get_output_embeddings())
                if module is not None for p in module.parameters()}
    matrices = [p for p in params if p.ndim == 2 and id(p) not in excluded]
    auxiliary = [p for p in params if p.ndim != 2 or id(p) in excluded]
    optimizers = []
    if matrices:
        optimizers.append(torch.optim.Muon(matrices, **kwargs))
    if auxiliary:
        optimizers.append(torch.optim.AdamW(auxiliary, **kwargs))
    return CombinedOptimizer(optimizers)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_questions(path: Path) -> list[Question]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            # #### 分隔的是答案中的推理与最终答案，不是 QA
            rationale = record["answer"]
            if "####" not in rationale:
                raise ValueError(f"{path}:{line_number}: answer 缺少 ####")
            answer = rationale.rsplit("####", 1)[1].strip()
            if not answer or not record["question"].strip():
                raise ValueError(f"{path}:{line_number}: 题目或最终答案为空")
            questions.append(Question(record["question"], answer))
    if not questions:
        raise ValueError(f"数据集为空：{path}")
    return questions


def load_prompt_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError("prompt 模板必须包含 {question}")
    template.format(question="检查占位符")
    return template


def build_runtime(config: TrainConfig) -> Runtime:
    # 1. 先将训练模型放到目标设备，再创建优化器
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=getattr(torch, config.dtype),
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须定义 EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
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
        )
        vllm_utils.wait_for_server(server.base_url, server.process, server.startup_timeout)
        server.init_weight_sync(str(device))
    except BaseException:
        server.stop()
        raise
    return Runtime(model, tokenizer, optimizer, server)


def sample_questions(questions, count, step, seed, strategy="random") -> list[Question]:
    if not 0 < count <= len(questions) or step < 1:
        raise ValueError("要求 0 < count <= 题目数 且 step >= 1")
    steps_per_epoch = len(questions) // count
    epoch, batch_index = divmod(step - 1, steps_per_epoch)
    indices = list(range(len(questions)))
    if strategy == "random":
        random.Random(seed + epoch).shuffle(indices)
    elif strategy != "order":
        raise ValueError(f"未知采样策略：{strategy}")
    start = batch_index * count
    return [questions[i] for i in indices[start:start + count]]


def sync_generator(runtime: Runtime, config: TrainConfig) -> None:
    if config.generation_backend == "vllm":
        runtime.generator.sync_policy_weights(runtime.model)


def generate_rollouts(runtime, questions, template, config) -> RolloutBatch:
    # 1. 同题的 group_size 个 prompt 连续排列
    prompts = [template.format(question=q.question) for q in questions for _ in range(config.group_size)]
    truths = [q.ground_truth for q in questions for _ in range(config.group_size)]
    responses = []
    if config.generation_backend == "vllm":
        completions = runtime.generator.generate_completions(prompts, {
            "temperature": config.temperature, "max_tokens": config.max_new_tokens,
            "n": 1, "seed": config.seed, "stop": ["</answer>"] if config.reward_type in ("r1_zero", "gsm8k_numeric") else None,
            "include_stop_str_in_output": True,
        }, batch_size=config.generation_batch_size)
        responses = [
            completion.text + (
                runtime.tokenizer.eos_token
                if completion.token_ids and completion.token_ids[-1] == runtime.tokenizer.eos_token_id
                else ""
            ) for completion in completions
        ]
    else:
        # 2. 左 padding 供 decoder-only 批量生成使用，显式传 attention_mask
        tokenizer = runtime.tokenizer
        previous_padding, previous_mode = tokenizer.padding_side, runtime.model.training
        tokenizer.padding_side = "left"
        runtime.model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(prompts), config.generation_batch_size):
                    inputs = tokenizer(prompts[start:start + config.generation_batch_size],
                                       padding=True, add_special_tokens=False, return_tensors="pt")
                    inputs = {key: value.to(runtime.model.device) for key, value in inputs.items()}
                    width = inputs["input_ids"].shape[1]
                    context_limit = getattr(runtime.model.config, "max_position_embeddings", None)
                    if context_limit and width + config.max_new_tokens > context_limit:
                        raise ValueError("prompt + max_new_tokens 超出模型上下文长度")
                    kwargs = dict(max_new_tokens=config.max_new_tokens, do_sample=config.temperature > 0,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    if config.temperature > 0:
                        kwargs.update(temperature=config.temperature, top_k=0, top_p=1.0)
                    generated = runtime.generator.generate(**inputs, **kwargs)
                    # 3. 只解码新 token，不把 prompt 当作回答
                    for row in generated[:, width:]:
                        text = tokenizer.decode(row, skip_special_tokens=True)
                        trimmed = config.reward_type in ("r1_zero", "gsm8k_numeric") and "</answer>" in text
                        if trimmed:
                            text = text.split("</answer>", 1)[0] + "</answer>"
                        # 只计入实际生成的 EOS，不给长度截断的回答伪造终止 token
                        if not trimmed and torch.any(row == tokenizer.eos_token_id):
                            text += tokenizer.eos_token
                        if not text:
                            raise ValueError("生成了无法解码的空回答，请检查 tokenizer 或生成参数")
                        responses.append(text)
        finally:
            tokenizer.padding_side = previous_padding
            runtime.model.train(previous_mode)
    if len(responses) != len(prompts):
        raise RuntimeError("生成回答数量与 prompt 数量不一致")
    return RolloutBatch(prompts, responses, truths)


def reward_function(config):
    if config.reward_type == "gsm8k_numeric":
        return numeric_reward
    try:
        from alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"作业评分器缺少依赖 {error.name}；可显式使用 --reward-type gsm8k_numeric，"
            "或在获得安装授权后配置原评分器依赖"
        ) from error
    return r1_zero_reward_fn if config.reward_type == "r1_zero" else question_only_reward_fn


def numeric_reward(response, ground_truth):
    """轻量数值评分，显式区别于原作业的数学等价评分器。"""
    # 只提取明确的最终答案位置，不用推理中最后一个数字猜测答案
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", response)
    answers = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    final = boxed[-1] if boxed else (answers[-1].strip() if answers else None)
    # r1 格式中的答案必须有正确的结束标签；boxed 位于未闭合推理中不算答案
    if "</think>" in response or "<answer>" in response:
        if "</think> <answer>" not in response or "</answer>" not in response:
            final = None
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    correct = False
    if final is not None:
        normalized = final.strip().replace(",", "")
        target = str(ground_truth).strip().replace(",", "")
        if re.fullmatch(number_pattern, normalized) and re.fullmatch(number_pattern, target):
            try:
                correct = Decimal(normalized) == Decimal(target)
            except InvalidOperation:
                pass
    return {"format_reward": float(final is not None), "answer_reward": float(correct), "reward": float(correct)}


def train_on_rollouts(runtime, rollouts, config):
    # eval 关闭 dropout，但仍保留梯度；让打分与生成使用同一策略
    previous_mode = runtime.model.training
    runtime.model.eval()
    try:
        return run_grpo_train_step(
            model=runtime.model, tokenizer=runtime.tokenizer, optimizer=runtime.optimizer,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm, reward_fn=reward_function(config),
            repeated_prompts=rollouts.repeated_prompts, rollout_responses=rollouts.responses,
            repeated_ground_truths=rollouts.repeated_ground_truths, group_size=config.group_size,
            importance_reweighting_method="none",
        )
    finally:
        runtime.model.train(previous_mode)


def evaluate(runtime, questions, template, config):
    # 固定验证随机状态；评测不消耗训练的采样随机序列
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            seed_everything(config.seed + 100000)
            eval_config = replace(config, group_size=1, temperature=0.0, seed=config.seed + 100000)
            rollouts = generate_rollouts(runtime, questions, template, eval_config)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    scores = [reward_function(config)(r, g) for r, g in zip(rollouts.responses, rollouts.repeated_ground_truths)]
    return {
        "mean_reward": sum(s["reward"] for s in scores) / len(scores),
        "accuracy": sum(s["answer_reward"] for s in scores) / len(scores),
        "format_rate": sum(s["format_reward"] for s in scores) / len(scores),
        "mean_response_tokens": sum(len(runtime.tokenizer.encode(r, add_special_tokens=False)) for r in rollouts.responses) / len(scores),
        "examples": [{"question": q.question, "ground_truth": q.ground_truth, "response": r, "score": s}
                     for q, r, s in zip(questions[:3], rollouts.responses[:3], scores[:3])],
    }


def json_value(value):
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def log_metrics(output_dir, step, split, metrics):
    record = json_value({"step": step, "split": split, **metrics})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "examples"}, ensure_ascii=False, allow_nan=False), flush=True)


def save_checkpoint(runtime, config, step):
    destination = config.output_dir / f"checkpoint-{step:06d}"
    destination.mkdir(parents=True, exist_ok=False)
    runtime.model.save_pretrained(destination)
    runtime.tokenizer.save_pretrained(destination)
    torch.save({"step": step, "optimizer": runtime.optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               destination / "training_state.pt")
    (destination / "config.json.run").write_text(json.dumps(json_value(asdict(config)), indent=2), encoding="utf-8")


def close_runtime(runtime):
    if isinstance(runtime.generator, vllm_utils.VLLMServer):
        runtime.generator.stop()


def train(config):
    # 1. 准备数据，拒绝覆盖已有实验结果
    seed_everything(config.seed)
    train_questions = load_questions(config.train_path)
    eval_questions = load_questions(config.eval_path)[:config.eval_examples]
    if len(train_questions) < config.prompts_per_batch:
        raise ValueError("训练题目数小于 prompts_per_batch")
    template = load_prompt_template(config.prompt_path)
    reward_function(config)  # 在加载大模型前检查评分器依赖
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError("输出目录非空，请选择新的 output-dir")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "run_config.json").write_text(json.dumps(json_value(asdict(config)), indent=2), encoding="utf-8")
    # 2. 初始化模型、优化器及生成后端
    runtime = build_runtime(config)
    try:
        sync_generator(runtime, config)
        log_metrics(config.output_dir, 0, "eval", evaluate(runtime, eval_questions, template, config))
        for step in range(1, config.num_steps + 1):
            # 3. 同步当前策略，生成本轮分组回答
            questions = sample_questions(train_questions, config.prompts_per_batch, step,
                                         config.seed, config.sampling_strategy)
            started = time.monotonic()
            sync_generator(runtime, config)
            rollouts = generate_rollouts(runtime, questions, template, replace(config, seed=config.seed + step))
            generation_seconds = time.monotonic() - started
            # 4. 本批 rollout 仅更新一次，分别记录生成和更新耗时
            started = time.monotonic()
            loss, metadata = train_on_rollouts(runtime, rollouts, config)
            if runtime.model.device.type == "cuda":
                torch.cuda.synchronize(runtime.model.device)
            log_metrics(config.output_dir, step, "train", {
                **metadata, "loss": loss, "generation_seconds": generation_seconds,
                "update_seconds": time.monotonic() - started,
                "examples": [{"prompt": p, "response": r, "ground_truth": g}
                             for p, r, g in zip(rollouts.repeated_prompts[:3], rollouts.responses[:3], rollouts.repeated_ground_truths[:3])],
            })
            # 5. 验证并保存最新策略
            if step % config.eval_every == 0 or step == config.num_steps:
                sync_generator(runtime, config)
                log_metrics(config.output_dir, step, "eval", evaluate(runtime, eval_questions, template, config))
            if step % config.save_every == 0 or step == config.num_steps:
                save_checkpoint(runtime, config, step)
    finally:
        close_runtime(runtime)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    for name, default in (("train-path", PROJECT_DIR / "data/gsm8k/train.jsonl"),
                          ("eval-path", PROJECT_DIR / "data/gsm8k/test.jsonl"),
                          ("prompt-path", PROJECT_DIR / "alignment/prompts/r1_zero.prompt"),
                          ("output-dir", PROJECT_DIR / "outputs/grpo")):
        parser.add_argument(f"--{name}", type=Path, default=default)
    for name, default in (("seed", 42), ("num-steps", 2), ("prompts-per-batch", 2), ("group-size", 2),
                          ("gradient-accumulation-steps", 2), ("max-new-tokens", 128), ("eval-every", 1),
                          ("eval-examples", 4), ("save-every", 1), ("generation-batch-size", 2),
                          ("vllm-port", 8000), ("vllm-gpu", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("learning-rate", 1e-5), ("max-grad-norm", 1.0), ("temperature", 1.0), ("weight-decay", 0.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--optimizer-name", choices=["adam", "adamw", "muon"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--sampling-strategy", choices=["random", "order"], default="random")
    parser.add_argument("--reward-type", choices=["gsm8k_numeric", "r1_zero", "question_only"], default="gsm8k_numeric")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    args = parser.parse_args()
    for name in ("num_steps", "prompts_per_batch", "group_size", "gradient_accumulation_steps", "max_new_tokens",
                 "eval_every", "eval_examples", "save_every", "generation_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.group_size < 2:
        parser.error("group_size must be at least 2")
    if (args.prompts_per_batch * args.group_size) % args.gradient_accumulation_steps:
        parser.error("rollout batch size must be divisible by gradient accumulation steps")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.temperature <= 0 or args.weight_decay < 0:
        parser.error("lr/max_grad_norm/temperature must be positive; weight_decay must be nonnegative")
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
