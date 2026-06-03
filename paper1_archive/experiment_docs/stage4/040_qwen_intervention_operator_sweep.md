# Stage4-040 Qwen Intervention Operator Sweep

## 目的

测试 V4 restore 失败是否是因为当前 decoder-vector add 干预算子太弱或不适合 Qwen。

## 输入

- V4 L14 primary/strict candidates
- paperpack72 images/masks
- Qwen2.5-VL-PLT

## 输出

- `stage4_qwen_native_route_operator_sweep_*`

## 方法

比较：

- decoder-vector add
- scaled decoder-vector add
- residual delta patch
- PLT topK reconstruction patch
- PLT reconstruction error patch

## 结果

待运行。

## 预期与实际偏差

如果 residual patch 成立但 feature patch 不成立，写 `PLT intervention/localization failure`。

## 结论

待运行。
