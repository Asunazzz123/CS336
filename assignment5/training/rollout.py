"""Application-level generation and policy synchronization."""
import torch
from training.types import RolloutBatch
from training.configs import RuntimeConfig as TrainConfig
from training.types import Runtime
from training.reproducibility import autocast_context


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
                    with autocast_context(runtime.model, config.autocast):
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
