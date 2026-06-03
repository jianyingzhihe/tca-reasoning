# Stage 2F Cross-model

本目录保存跨模型验证相关材料。

当前定位：

```text
Stage 2F 是 feasibility / mini replication，不是当前主线。
它进入 Stage 2，但不抢 Stage 2A targeted replication。
```

执行顺序：

1. `cross_model_candidate_table.csv`：公开资产和候选模型表；
2. `cross_model_loader_smoke.md`：loader / hook / processor / transcoder format smoke；
3. `cross_model_one_case_readout.md`：单样本 attribution / intervention smoke；
4. `cross_model_region_replication_readout.md`：3-5 case mini region replication。

当前最高优先级候选：

```text
KokosDev/qwen2p5vl-7b-clt
```

原因：

1. 它是 VLM candidate；
2. 有明确 base model；
3. 有 `config.yaml`；
4. 有 per-layer `safetensors`；
5. 比 LLaVA 候选更接近当前 transcoder set 格式。

