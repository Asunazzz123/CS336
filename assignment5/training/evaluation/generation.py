"""Generated-answer evaluation, separate from training rewards and updates."""
import random
from dataclasses import replace
import numpy as np
import torch
from alignment.rewards.factory import reward_function
from training.reproducibility import seed_everything
from training.rollout import generate_rollouts


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
