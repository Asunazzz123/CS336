# Engine
- 双卡 FSDP SFT, GPU adamw
- Mixed Precision
- GPU Adamw (In which situation, 单卡显存不足，GPU分担训练状态，跨卡sharding)

- 训推混合策略与训推分离策略，Rollout 推理卡和 训练卡的负载均衡和训练卡的KVCache的空间局部性矛盾，动态平衡时拓扑结构变化
- Weight Resharding -- FSDP内部，训练rollout switching.

# Training
- RL optimizer Adamw or Muon

