# Stage4-036 Qwen Layer-14 Source-Tracing Results

## 目的

记录 Qwen layer 14 source-tracing rerun，替代此前 layer 26-heavy 的 source-tracing 负向证据。

## 输入

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `KokosDev/qwen2p5vl-7b-plt`
- layer `14`
- `paperpack72_primary` / `paperpack72_strict`

## 输出

- L14 graph metadata
- controlled compare rows
- node tables
- intervention rows
- decision artifacts

## 方法

source-tracing 参数：

- `--layer 14`
- `--candidate-pool-size 8192`
- `--max-feature-nodes 128`
- `--compare-topk-per-node 8`
- `--top-features-per-sample 8`
- `--position-filter visual_answer`

## 结果

### Primary Smoke

命令参数：

- layer: `14`
- candidate pool: `8192`
- max feature nodes: `128`
- compare topK per node: `8`
- top features per sample: `8`
- position filter: `visual_answer`

结果：

- A/D_visual_only graphs: `3/3`
- B/B_direct graphs: `3/3`
- matched valid samples: `3`
- controlled compare: produced
- node/edge tables: produced
- intervention rows: produced
- decision artifact: produced

Artifact prefix:

`stage4_qwen_source_tracing_primary_smoke_adapter_v4_L14`

### Primary Full

Primary full 已完成并拉回本地。

Artifacts prefix:

`stage4_qwen_source_tracing_primary_full_adapter_v4_L14`

结果：

- meta A rows: `72`
- meta B rows: `72`
- graph ok A: `70`
- graph ok B: `70`
- prompt-runs ok: `140`
- graph success rate: `0.9722`
- valid samples: `70`
- compare rows: `70`
- feature node rows: `1120`
- intervention rows: `2240`
- mean delta target logit: `-0.0051`
- fraction negative delta target logit: `0.3147`
- fraction rank hurt: `0.1170`
- best zeroing mode: `subtract`

Decision:

`qwen_source_tracing_not_supported`

## 预期与实际偏差

L14 source tracing 跑通，且 graph/compare 覆盖率高；但自动 source-tracing intervention 本身仍未达到 source route 支持标准。这说明单独使用当前 answer-aligned source-tracing node selection 仍未闭合。

这不是 Qwen hidden route 的负结论，因为 Stage4-033 已确认 hidden layer 14 route；它说明需要 Adapter V4 用 evidence specificity 和 zeroing damage 对自动节点重新筛选。

## 结论

Smoke 通过。L14 source-tracing runner、graph schema、controlled compare 和 intervention smoke 均可执行。该结果只证明工程链路可用，不作为机制正/负结论。
