# Remote Commands：Stage 2A Pretrace Top24

日期：2026-05-19  
用途：在服务器上跑 Stage 2A pretrace queue 的 B/D eval 和 answer-aligned trace。

本文件是假定服务器项目根目录为：

```text
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
```

需要先把本地这些文件同步到服务器同一相对位置或你指定的位置：

```text
doc/experiments/stage2/stage2a_candidate_selection/manifest_B_direct_stage2a_pretrace_top24.csv
doc/experiments/stage2/stage2a_candidate_selection/manifest_D_visual_only_stage2a_pretrace_top24.csv
doc/experiments/stage2/stage2a_candidate_selection/stage2a_trace_selected_ids_top24.csv
```

---

## 1. 服务器目录准备

```bash
cd /root/autodl-tmp/tca-reasoning/circuit_tracer_vlm

export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"

RUN_ROOT="outputs/phase_ab/ab_answer_aligned/stage2a_pretrace_top24_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"/{eval,pt_slotA_D_visual_only,pt_slotB_B_direct,logs}
```

---

## 2. 跑 B/D eval

`B_direct`：

```bash
python scripts/research/run_batch_eval.py \
  --manifest doc/experiments/stage2/stage2a_candidate_selection/manifest_B_direct_stage2a_pretrace_top24.csv \
  --output-csv "$RUN_ROOT/eval_B_direct.csv" \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 16 \
  --log-every 5 \
  2>&1 | tee "$RUN_ROOT/logs/eval_B_direct.log"
```

`D_visual_only`：

```bash
python scripts/research/run_batch_eval.py \
  --manifest doc/experiments/stage2/stage2a_candidate_selection/manifest_D_visual_only_stage2a_pretrace_top24.csv \
  --output-csv "$RUN_ROOT/eval_D_visual_only.csv" \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 16 \
  --log-every 5 \
  2>&1 | tee "$RUN_ROOT/logs/eval_D_visual_only.log"
```

---

## 3. 跑 answer-aligned trace

Slot B：`B_direct`

```bash
python scripts/research/run_batch_answer_aligned_attribute.py \
  --eval-csv "$RUN_ROOT/eval_B_direct.csv" \
  --selected-csv doc/experiments/stage2/stage2a_candidate_selection/stage2a_trace_selected_ids_top24.csv \
  --output-dir "$RUN_ROOT/pt_slotB_B_direct" \
  --metadata-csv "$RUN_ROOT/answer_aligned_meta_b.csv" \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 64 \
  --batch-size 1 \
  --offload disk \
  --answer-source predicted \
  2>&1 | tee "$RUN_ROOT/logs/trace_B_direct.log"
```

Slot A：`D_visual_only`

```bash
python scripts/research/run_batch_answer_aligned_attribute.py \
  --eval-csv "$RUN_ROOT/eval_D_visual_only.csv" \
  --selected-csv doc/experiments/stage2/stage2a_candidate_selection/stage2a_trace_selected_ids_top24.csv \
  --output-dir "$RUN_ROOT/pt_slotA_D_visual_only" \
  --metadata-csv "$RUN_ROOT/answer_aligned_meta_a.csv" \
  --transcoder-set tianhux2/gemma3-4b-it-plt \
  --dtype bfloat16 \
  --max-feature-nodes 64 \
  --batch-size 1 \
  --offload disk \
  --answer-source predicted \
  2>&1 | tee "$RUN_ROOT/logs/trace_D_visual_only.log"
```

---

## 4. 回传文件

完成后需要同步回本地：

```text
$RUN_ROOT/eval_B_direct.csv
$RUN_ROOT/eval_D_visual_only.csv
$RUN_ROOT/answer_aligned_meta_a.csv
$RUN_ROOT/answer_aligned_meta_b.csv
$RUN_ROOT/pt_slotA_D_visual_only/*.pt
$RUN_ROOT/pt_slotB_B_direct/*.pt
$RUN_ROOT/logs/*.log
```

建议本地落点：

```text
remote_sync/2026-05-19_stage2a_pretrace_top24/
```

---

## 5. 跑完后的本地分析步骤

回传后继续：

```text
1. build clean-core summary
2. run support/suppressor zeroing intervention smoke
3. run nearest matched control
4. run random4 matched control if needed
5. produce stage2a_pretrace_readout.md
6. select 10-15 region replication samples
```

---

## 6. 成功标准

Stage 2A-1 成功：

```text
24 个样本中至少 10-15 个 trace 成功；
至少 8 个样本 clean-core 可用；
至少 8 个样本出现 support source；
至少 6 个样本可构造 nearest control；
已有 mask 的样本占多数。
```

Stage 2A-1 partial：

```text
trace 成功，但 clean-core 或 nearest 不足；
需要再补一个 symbol_text_reading-heavy queue。
```

Stage 2A-1 fail：

```text
trace 大面积失败；
或者 eval 输出格式/answer prefix 不稳定；
或者 top24 大多没有 support route。
```

