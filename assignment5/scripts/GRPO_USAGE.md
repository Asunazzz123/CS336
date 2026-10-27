# GRPO 训练脚本使用说明

## 推荐使用方式

**强烈推荐使用 `train_grpo_unified.py`**，它整合了两个原始实现的所有功能，并修正了 `train_grpo_full.py` 的设计缺陷。

## 三个脚本说明

### 1. train_grpo_unified.py（推荐）✨

统一实现，支持单次更新和多 epoch PPO-style 训练。

```bash
# 快速单次更新
python scripts/train_grpo_unified.py \
  --model-id Qwen/Qwen3.5-4B-Base \
  --device cuda:0 \
  --num-epochs-per-rollout 1 \
  --output-dir outputs/quick

# 标准多 epoch（推荐）
python scripts/train_grpo_unified.py \
  --model-id Qwen/Qwen3.5-4B-Base \
  --device cuda:0 \
  --dtype bfloat16 \
  --num-epochs-per-rollout 3 \
  --output-dir outputs/standard
```

### 2. train_grpo.py（原始基础版）

纯 on-policy 单次更新，无 ratio clipping。

```bash
python scripts/train_grpo.py \
  --model-id Qwen/Qwen3.5-4B-Base \
  --device cuda:0 \
  --output-dir outputs/basic
```

**等价于：**
```bash
python scripts/train_grpo_unified.py \
  --num-epochs-per-rollout 1 \
  --advantage-normalizer none \
  --loss-normalization constant \
  [其他参数]
```

### 3. train_grpo_full.py（原始完整版）⚠️

**注意：** 此脚本存在设计缺陷，默认只做单 epoch 更新导致 ratio≈1，clip 不生效。

```bash
python scripts/train_grpo_full.py \
  --model-id Qwen/Qwen3.5-4B-Base \
  --device cuda:0 \
  --output-dir outputs/full
```

**推荐替代方案：**
```bash
python scripts/train_grpo_unified.py \
  --num-epochs-per-rollout 3 \
  --importance-reweighting-method grpo \
  --baseline mean \
  --advantage-normalizer std \
  --loss-normalization sequence \
  [其他参数]
```

## 快速决策指南

**选择脚本：**

| 场景 | 推荐脚本 | 配置 |
|------|---------|------|
| 新项目 | train_grpo_unified.py | num_epochs=3 |
| 快速实验 | train_grpo_unified.py | num_epochs=1 |
| 维护旧项目（基础版） | train_grpo.py | 保持原参数 |
| 维护旧项目（完整版） | train_grpo_unified.py | num_epochs=3（修正） |

**关键参数：`--num-epochs-per-rollout`**

- `1`: 单次更新，最快，适合快速迭代
- `3`: 标准 PPO-style，样本高效，适合正式训练（**推荐**）
- `5`: 高样本利用，适合数据稀缺场景

## 完整文档

详细文档位于 `docs/` 目录：

1. **[docs/grpo_quick_reference.md](docs/grpo_quick_reference.md)** - 快速参考
2. **[docs/grpo_unified_usage.md](docs/grpo_unified_usage.md)** - 详细使用手册
3. **[docs/grpo_comparison.md](docs/grpo_comparison.md)** - 技术对比分析
4. **[docs/grpo_migration_guide.md](docs/grpo_migration_guide.md)** - 迁移指南
5. **[docs/README.md](docs/README.md)** - 文档索引

## 常见问题

### Q: 我应该用哪个脚本？

**A:** 新项目直接用 `train_grpo_unified.py`，它功能最全且修正了已知问题。

### Q: train_grpo_full.py 有什么问题？

**A:** 它每步只做单次更新，导致 old_log_probs 和 new_log_probs 用相同参数计算，ratio≈1，clip 不生效。unified 版本通过 `--num-epochs-per-rollout 3` 修正了这个问题。

### Q: 如何复现论文结果？

**A:** 大多数 GRPO/PPO 论文使用 3-5 个 epoch：

```bash
python scripts/train_grpo_unified.py \
  --model-id <MODEL> \
  --num-epochs-per-rollout 3 \
  --cliprange 0.2 \
  [其他论文参数]
```

### Q: 显存不足怎么办？

**A:** 使用显存优化组合：

```bash
python scripts/train_grpo_unified.py \
  --gradient-checkpointing \
  --optimizer-device cpu \
  --prompts-per-batch 4 \
  --group-size 4 \
  [其他参数]
```

## 获取帮助

```bash
# 查看所有参数
python scripts/train_grpo_unified.py --help

# 查看文档
cat docs/README.md
```

## 总结

- ✅ 使用 `train_grpo_unified.py` 获得最佳体验
- ✅ 设置 `--num-epochs-per-rollout 3` 用于标准训练
- ✅ 查阅 `docs/` 目录获取详细文档
- ⚠️ 避免直接使用 `train_grpo_full.py`（除非维护旧项目）
