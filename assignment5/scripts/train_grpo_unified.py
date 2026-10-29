"""GRPO CLI: single/multi-epoch, Transformers/FSDP2."""
import argparse
import logging
from pathlib import Path
import sys
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from training.configs import UnifiedGRPOConfig
from training.validation import validate_config


def parse_args() -> UnifiedGRPOConfig:
    """统一参数解析，整合两个脚本的所有选项。"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # === 模型与数据 ===
    parser.add_argument("--model-id", required=True, help="HuggingFace 模型 ID")
    parser.add_argument("--train-path", type=Path, default=PROJECT_DIR / "data/gsm8k/train.jsonl")
    parser.add_argument("--eval-path", type=Path, default=PROJECT_DIR / "data/gsm8k/test.jsonl")
    parser.add_argument("--prompt-path", type=Path, default=PROJECT_DIR / "alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/grpo_unified")

    # === 设备与精度 ===
    parser.add_argument("--device", default="cpu", help="训练设备（cpu/cuda/cuda:0）")
    parser.add_argument("--training-backend", choices=["transformers", "fsdp2"], default="transformers",
                       help="fsdp2 requires torchrun with >=2 NVIDIA GPUs; device comes from LOCAL_RANK")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--autocast", choices=["none", "bfloat16"], default="none",
                       help="FP32 主权重 + BF16 autocast 计算")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="以重算换显存")

    # === 生成后端 ===
    parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--rollout-checkpoint-dir", type=Path, default=Path("/dev/shm/cs336-rollout"))
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-startup-timeout", type=int, default=600)

    # === 训练超参数 ===
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=100, help="训练步数")
    parser.add_argument("--prompts-per-batch", type=int, default=8, help="每批采样的题目数")
    parser.add_argument("--group-size", type=int, default=8, help="每题生成的回答数")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer-name", choices=["adam", "adamw", "muon"], default="adamw")
    parser.add_argument("--optimizer-device", choices=["model", "cpu"], default="model",
                       help="cpu 将优化器状态放在 CPU")

    # === 生成参数 ===
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--sampling-strategy", choices=["random", "order"], default="random")
    parser.add_argument("--reward-type", choices=["gsm8k_numeric", "r1_zero", "question_only"],
                       default="gsm8k_numeric")

    # === 多 epoch 与裁剪控制（核心融合参数）===
    parser.add_argument("--num-epochs-per-rollout", type=int, default=1,
                       help="每批 rollout 的更新 epoch 数。1=单次更新（快速），>1=多 epoch PPO-style（clip 生效）")
    parser.add_argument("--importance-reweighting-method",
                       choices=["none", "noclip", "grpo", "gspo"], default="grpo",
                       help="重要性加权方法。grpo=token-level clip，gspo=sequence-level clip")
    parser.add_argument("--cliprange", type=float, default=0.2,
                       help="PPO clip 范围（epsilon）")

    # === 奖励归一化 ===
    parser.add_argument("--baseline", choices=["mean", "none"], default="mean",
                       help="优势基线。mean=减组均值，none=不减")
    parser.add_argument("--advantage-normalizer", choices=["std", "none", "mean"], default="std",
                       help="优势归一化。std=除以标准差，none=不归一化，mean=除以均值")
    parser.add_argument("--loss-normalization", choices=["sequence", "constant"], default="sequence",
                       help="损失归一化。sequence=先逐序列平均再批平均，constant=按 token 总数平均")

    # === 评估与保存 ===
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--checkpoint-format", choices=["dcp", "hf"], default="dcp",
                       help="FSDP periodic output format; best checkpoints always use HF")
    parser.add_argument("--no-save-optimizer-state", dest="save_optimizer_state",
                       action="store_false",
                       help="checkpoint 不保存优化器状态")
    parser.add_argument("--best-checkpoint-dir", type=Path,
                       help="滚动最佳 model-only 缓存目录（建议 /dev/shm）")

    args = parser.parse_args()

    config = UnifiedGRPOConfig(**vars(args))
    try:
        validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    if config.best_checkpoint_dir is not None:
        config.save_optimizer_state = False
    return config


def main():
    config = parse_args()
    from training.trainers.grpo import train
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    train(config)


if __name__ == "__main__":
    main()
