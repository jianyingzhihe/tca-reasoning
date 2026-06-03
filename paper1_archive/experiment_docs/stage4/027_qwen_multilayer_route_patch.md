# Stage4-024 Qwen Multi-Layer Route Patch

## 目的

测试 Qwen 的路线是否分散在多层，导致单层 feature/hidden patch 过弱。

## 输入

- Hidden lattice top layers
- PLT layer sweep candidates
- paperpack72 primary / strict

## 输出

- `stage4_qwen_decisive_route_multilayer_*`

## 方法

从 hidden lattice 中选择 top layer groups，例如 `20+24+26` 或 `22+24+28`，同时 patch 多层 PLT feature groups。仍使用 source/control、wrong-target、shifted/shuffled controls。

## 结果

待 hidden lattice 与 PLT layer sweep 完成后运行。

## 预期与实际偏差

如果单层失败、多层通过，写 `Qwen route is distributed / multi-layer`。如果多层仍失败，不能写 Qwen 没有机制，只能写当前多层 PLT route patch 不支持 Gemma-style sparse route。

## 结论

待运行。
