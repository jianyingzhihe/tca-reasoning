# Stage4-008 LLaVA-CLT Layer/TopK Diagnostic

## 目的

判断 LLaVA-CLT feature/source route 未证成是否来自 layer/topK 选择，而不是直接下“没有机制”的负结论。

## 输入

- LLaVA-1.5-7B base。
- `KokosDev/llava15-7b-clt`。
- paperpack72 primary/strict。
- layer sweep: `12,15,18,21`。
- topK: `1,4,8,16,32`。

## 结果

待运行。

## 结论

待运行。若所有 sweep 失败，结论是 `feature/source route not established under tested CLT layers/controls`。

