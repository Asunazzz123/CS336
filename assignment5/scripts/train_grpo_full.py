"""Full on-policy GRPO: 多 epoch PPO-style 更新的包装器。

此脚本已迁移为 train_grpo_unified.py 的包装器，保持向后兼容。
所有参数会自动转发到 unified 版本，并设置：
- num_epochs_per_rollout=3 (多 epoch 更新，修正原实现的缺陷)
- importance_reweighting_method=grpo (token-level clip)
- baseline=mean (默认)
- advantage_normalizer=std (默认)
- loss_normalization=sequence (默认)

如需更多控制，请直接使用 train_grpo_unified.py。
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    """包装器主函数：转发到 train_grpo_unified.py。"""
    # 构造 unified 脚本路径
    unified_script = PROJECT_DIR / "scripts" / "train_grpo_unified.py"

    if not unified_script.exists():
        print(f"错误：找不到 {unified_script}", file=sys.stderr)
        print("请确保 train_grpo_unified.py 存在于 scripts/ 目录", file=sys.stderr)
        sys.exit(1)

    # 转发所有参数，并强制设置兼容参数
    args = sys.argv[1:]  # 移除脚本名

    # 添加默认兼容参数（如果用户未指定）
    compat_args = []
    args_str = " ".join(args)

    if "--num-epochs-per-rollout" not in args_str:
        compat_args.extend(["--num-epochs-per-rollout", "3"])
    if "--importance-reweighting-method" not in args_str:
        compat_args.extend(["--importance-reweighting-method", "grpo"])
    if "--baseline" not in args_str:
        compat_args.extend(["--baseline", "mean"])
    if "--advantage-normalizer" not in args_str:
        compat_args.extend(["--advantage-normalizer", "std"])
    if "--loss-normalization" not in args_str:
        compat_args.extend(["--loss-normalization", "sequence"])

    # 执行 unified 脚本
    cmd = [sys.executable, str(unified_script)] + args + compat_args

    print(f"[train_grpo_full.py] 转发到 train_grpo_unified.py (多 epoch 模式)", file=sys.stderr)
    print(f"[train_grpo_full.py] 默认 num_epochs_per_rollout=3 (修正原版缺陷)", file=sys.stderr)
    print(f"[train_grpo_full.py] 等效命令: {' '.join(cmd)}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"执行失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
