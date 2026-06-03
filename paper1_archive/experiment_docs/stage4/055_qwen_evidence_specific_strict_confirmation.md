# Stage4-055 Qwen Evidence-Specific Strict Confirmation

## 目的

用 strict paperpack 复验 primary 选出的 evidence-specific nodes，防止 post-hoc 挑节点。

## 方法

- strict 不重新筛 feature、layer、position、阈值。
- 复用 primary manifest 中的 exact layer/position/feature。
- 若 strict prompt-run metadata 可用，则替换 question/image/mask/answer 字段后运行同一 targeted validation。

## 判据

- primary 与 strict 同向通过 source/control、correct/wrong，可写 causal evidence-specific nodes support。
- 若 strict 的 mask restore 也满足 real > shifted/shuffled，可升级为 evidence-linked route support。
- 若 primary 有信号但 strict 不稳定，只写 primary targeted diagnostic。

## 当前状态

Strict manifest 已生成，候选规则冻结；strict full 等 primary targeted full 完成后再启动。Strict 缺失 2 条 prompt-run，因此 strict pool rows 为 2145，main rows 仍为 160。
