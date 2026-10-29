"""SFT CLI; application logic lives in training.trainers.sft."""
import argparse
from pathlib import Path
import sys
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from training.configs import SFTConfig
from training.validation import validate_sft_config as _validate_config


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--train-path", type=Path, default=PROJECT_DIR / "data/gsm8k/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/sft")
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-backend", choices=["transformers", "fsdp2"], default="transformers")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--optimizer-name", choices=["adam", "adamw", "muon"], default="adamw")
    parser.add_argument("--optimizer-device", choices=["model", "cpu"], default="model",
                        help="cpu 将 Adam/AdamW 的 FP32 主权重与优化器状态放在 CPU")
    parser.add_argument("--response-format", choices=["r1_zero", "boxed"], default="r1_zero")
    parser.add_argument("--max-steps", type=int)
    for name, default in (("seed", 42), ("num-epochs", 1), ("batch-size", 1),
                          ("gradient-accumulation-steps", 8), ("max-seq-length", 1024),
                          ("eval-every", 50), ("save-every", 100), ("eval-examples", 64),
                          ("eval-generation-examples", 0), ("max-new-tokens", 512),
                          ("generation-batch-size", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("validation-fraction", 0.1), ("learning-rate", 1e-5),
                          ("max-grad-norm", 1.0), ("weight-decay", 0.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--autocast", choices=["none", "bfloat16"], default="none",
                        help="建议配合 --dtype float32：FP32 主权重 + BF16 autocast 计算")
    parser.add_argument("--no-save-optimizer-state", dest="save_optimizer_state", action="store_false",
                        help="checkpoint 不保存优化器状态（磁盘紧张时用）")
    parser.add_argument("--best-checkpoint-dir", type=Path,
                        help="滚动最佳model-only缓存目录（建议/dev/shm）；结束仅保存checkpoint-best")
    parser.add_argument("--prepare-only", action="store_true")
    config = SFTConfig(**vars(parser.parse_args()))
    if config.best_checkpoint_dir is not None:
        config.save_optimizer_state = False
    try:
        if config.optimizer_device == "cpu" and config.optimizer_name == "muon":
            raise ValueError("CPU optimizer offload only supports adam/adamw")
        _validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    return config


def main():
    config = parse_args()
    from training.trainers.sft import train
    print(train(config))


if __name__ == "__main__":
    main()
