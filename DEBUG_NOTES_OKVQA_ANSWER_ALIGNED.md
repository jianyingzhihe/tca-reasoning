# OK-VQA / Answer-Aligned Attribution Debug Notes

## Purpose

This file records the main bugs we hit while moving from:

- plain OK-VQA prompt evaluation
- to prompt-level circuit tracing
- to **answer-token-aligned** circuit tracing

The goal is to prevent future windows/sessions from rediscovering the same problems.

This document is intentionally detailed. It focuses on:

- what broke
- what symptom we saw
- why it happened
- how we fixed it
- what commands are useful to verify the fix

---

## Repo / Path Context

### Local repo root

`D:\code\Bridging\vlm-circuit-tracing`

### Project subdirectory

`D:\code\Bridging\vlm-circuit-tracing\circuit_tracer_vlm`

### New server repo root

`/root/autodl-tmp/tca-reasoning`

### New server project dir

`/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm`

### New server data root

`/root/autodl-tmp/tca-reasoning/data`

### Old server paths that appeared in old CSVs

Examples:

- `/home/xtyu/tca-reasoning/...`
- `/home/xtyu/tca-reasoning/data/okvqa/images/...`

Those old paths caused many migration failures when reused on the new machine.

---

## High-Level Timeline

We started from a setup where:

- prompt A / B / D evaluation CSVs already existed
- A/B bucket CSV had already been built locally
- we wanted to compare **answer token** circuits, not just prompt-start circuits

Then we moved the workflow onto a new 48 GB server and hit several categories of bugs:

1. environment/bootstrap bugs
2. path/data layout bugs
3. interpreter / package resolution bugs
4. multimodal prompt formatting bugs
5. answer-token alignment bugs
6. process/memory/GPU bugs

---

## Important Conceptual Clarification

### Old prompt-level attribution

The original attribute path effectively analyzed:

- given prompt + image
- what is the **next token** the model wants to generate

This is useful, but it is **not** the same as analyzing the internal circuit for the final answer token.

### New answer-aligned attribution

The new path is intended to analyze:

- the prompt
- plus the already-generated assistant prefix, e.g. `The answer is `
- then target the **first answer token**

Example:

- generated text: `The answer is bench.`
- assistant prefix: `The answer is `
- target token: first token in `bench`

This is the correct direction if we want to compare A-vs-B at the answer position.

---

## Bug 1: `dev.sh` / `load_env.sh` polluted the current shell

### Symptom

After running:

```bash
source scripts/server/dev.sh
```

or sometimes after sourcing env helpers, later commands would fail and the terminal session would appear to close or the shell would die after any non-zero exit.

We also saw errors like:

```bash
-bash: debian_chroot: unbound variable
```

### Root cause

Old environment helper scripts used:

```bash
set -euo pipefail
```

When sourced into an interactive shell, that setting leaked into the shell itself.

So later:

- any failing command
- any unbound variable
- or some shell startup code in `.bashrc`

could immediately kill the shell or tab.

### Fix

We stopped relying on `dev.sh` for normal usage and used explicit activation instead:

```bash
cd /root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
source .venv/bin/activate
source scripts/server/load_env.sh .env
```

We also refactored local `dev.sh` earlier to avoid this behavior, but the safest operational advice is:

- do not depend on `source dev.sh` for critical workflow steps
- prefer explicit `source .venv/bin/activate`
- prefer explicit `source scripts/server/load_env.sh .env`

### Verification

```bash
which python
python -c "import sys; print(sys.executable)"
```

---

## Bug 2: `.env` parsing bugs

### Symptom

Some variables loaded while others did not.

For example:

- `HF_HOME` existed
- `HF_TOKEN` existed
- `HF_ENDPOINT` was empty

### Root causes

#### 2.1 Last line missing trailing newline

The `.env` parser loop in `load_env.sh` could miss the final line if the file ended without a newline.

This hit `HF_ENDPOINT` specifically.

#### 2.2 Use of `$HOME` inside `.env`

We also saw values like:

```bash
HF_HOME=$HOME/tca-reasoning/data/hf_cache
```

Sometimes this was not expanded the way we wanted.

### Fix

Use explicit absolute paths and rewrite `.env` cleanly.

Recommended style:

```bash
HF_TOKEN=...
HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub
HF_ENDPOINT=https://hf-mirror.com
```

Ensure the file ends with a trailing newline.

### Verification

```bash
source scripts/server/load_env.sh .env
echo "$HF_TOKEN"
echo "$HF_HOME"
echo "$HF_ENDPOINT"
```

---

## Bug 3: `(.venv)` prompt did not mean the correct Python was actually active

### Symptom

Shell prompt showed `(.venv)` but:

```bash
which python
which pip
```

still pointed to:

- `/root/miniconda3/bin/python`
- `/root/miniconda3/bin/pip`

### Root cause

The shell prompt alone is not authoritative. Conda base and venv state were mixed.

### Fix

Always verify the real interpreter explicitly.

Preferred commands:

```bash
which python
python -c "import sys; print(sys.executable)"
which pip
python -m pip --version
```

When installing into the project venv, use the venv python explicitly:

```bash
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python -m pip install ...
```

### Verification

The correct path should be:

```bash
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python
```

---

## Bug 4: `check_hf_access.py` did not download full weights

### Symptom

`check_hf_access.py` succeeded, but later model runs still triggered more downloads or failed unexpectedly.

### Root cause

That script only checks small config files, not full model weights.

### Fix / Clarification

Treat `check_hf_access.py` as:

- access test
- mirror test
- cache path sanity test

not as proof that full model weights are fully cached.

Full weights are actually exercised by:

- `run_gemma_smoke.sh`
- `circuit-tracer attribute`
- eval scripts that load the model

---

## Bug 5: Old CSVs contained old server image paths

### Symptom

Answer-aligned runs failed with image-path errors.

### Root cause

Old files like:

- `promptA_eval_final.csv`
- `promptB_eval.csv`
- `ab_buckets_by_hit_from_latest.csv`

had `image_path` fields pointing to the old machine:

```bash
/home/xtyu/tca-reasoning/data/okvqa/images/...
```

### Fix

On the new server, we rewrote them to:

```bash
/root/autodl-tmp/tca-reasoning/data/okvqa/images/...
```

### Verification

```bash
python - <<'PY'
import csv
for p in [
    "research/work/promptA_eval_final.csv",
    "research/work/promptB_eval.csv",
    "research/work/ab_buckets_by_hit_from_latest.csv",
]:
    row = next(csv.DictReader(open(p, encoding="utf-8")))
    print(p, "->", row["image_path"])
PY
```

---

## Bug 6: OK-VQA data existed but was not extracted into the expected structure

### Symptom

The new server had:

```bash
/root/autodl-tmp/tca-reasoning/data/okvqa/raw
```

but not:

```bash
/root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014
```

So image loads failed even after path rewriting.

### Root cause

The dataset package had only been downloaded, not extracted.

### Fix

Extract into the expected structure:

```bash
cd /root/autodl-tmp/tca-reasoning/data/okvqa
mkdir -p annotations questions images

unzip -o raw/mscoco_val2014_annotations.json.zip -d annotations
unzip -o raw/OpenEnded_mscoco_val2014_questions.json.zip -d questions
unzip -q raw/val2014.zip -d images
```

### Verification

```bash
ls /root/autodl-tmp/tca-reasoning/data/okvqa/annotations
ls /root/autodl-tmp/tca-reasoning/data/okvqa/questions
find /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014 -name "*.jpg" | wc -l
```

Expected JPG count:

```bash
40504
```

---

## Bug 7: answer-aligned prompt initially omitted image token

### Symptom

We saw:

```text
ValueError: Prompt contained 0 image tokens but received 1 images.
```

### Root cause

The initial answer-aligned script passed:

```python
question
```

instead of:

```python
f"<start_of_image> {question}"
```

### Fix

Update answer-aligned command construction to include `<start_of_image>`.

Affected file:

- [run_batch_answer_aligned_attribute.py](D:/code/Bridging/vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_batch_answer_aligned_attribute.py)

Git commit that fixed this:

- `2bea640 fix answer-aligned prompt image token`

---

## Bug 8: external command resolution pointed to the wrong environment

### Symptom

We saw logs like:

```bash
/root/miniconda3/bin/python -m circuit_tracer ...
```

instead of the project venv.

### Root cause

Relying on PATH / shell state was not robust enough.

### Fixes

Several improvements were applied:

1. force answer-attribution code to use project venv python
2. propagate `TCA_PYTHON`
3. make the outer bash script use a fixed Python path

Important commits in this area:

- `1356e8c use current python for answer-aligned attribution`
- `eeb1d62 pin answer-aligned pipeline to venv python`

---

## Bug 9: direct package import bypassed vendored TransformerLens

### Symptom

We saw:

```text
ImportError: cannot import name 'HookedVLTransformer' from 'transformer_lens'
```

and the imported path pointed to:

```bash
/root/miniconda3/lib/python3.10/site-packages/transformer_lens/...
```

### Root cause

The vendored `third_party/TransformerLens` path insertion existed in:

- `circuit_tracer/__main__.py`

but answer-aligned code started importing:

```python
from circuit_tracer import ReplacementModel
```

which bypassed that old entrypoint-specific path injection.

### Fix

Move the vendored `TransformerLens` preference logic into package init:

- [circuit_tracer/__init__.py](D:/code/Bridging/vlm-circuit-tracing/circuit_tracer_vlm/circuit_tracer/__init__.py)

Git commit:

- `f312b16 prefer vendored transformer lens in package init`

### Important verification note

Do **not** test this by doing:

```python
import transformer_lens
from circuit_tracer import ReplacementModel
```

because importing `transformer_lens` first can already cache the wrong module in `sys.modules`.

The better test is:

```bash
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python - <<'PY'
from circuit_tracer import ReplacementModel
import transformer_lens
print("transformer_lens:", transformer_lens.__file__)
print("ReplacementModel import ok")
PY
```

---

## Bug 10: `.venv` itself was incomplete

### Symptom

We saw errors like:

```text
ModuleNotFoundError: No module named 'huggingface_hub'
ModuleNotFoundError: No module named 'transformers'
```

### Root cause

The venv either:

- had never been fully initialized
- or the shell was still pointing to conda Python even though the prompt showed `(.venv)`

### Fix

Use the venv python explicitly to install packages.

Example:

```bash
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python -m pip install --upgrade pip
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python -m pip install \
  "torch>=2.0.0" \
  "transformers>=4.57.0,<5.0.0" \
  "transformer-lens==2.18.0" \
  "huggingface_hub>=0.26.0" \
  "pydantic>=2.0.0" \
  "safetensors>=0.5.0" \
  "tokenizers>=0.21.0" \
  "tqdm>=4.60.0" \
  "einops>=0.8.0" \
  "numpy>=1.24.0,<2.0.0" \
  "pyyaml>=5.1" \
  "requests>=2.28.0" \
  "pillow>=10.0.0"
```

### Verification

```bash
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python -c "import transformers, torch, einops, huggingface_hub; print('env ok')"
```

---

## Bug 11: Why old `run_batch_attribute.py` seemed to work

### Important clarification

The old working transcoder batch path was:

- [run_batch_attribute.py](D:/code/Bridging/vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_batch_attribute.py)

This script is **not** a long-lived single-process reuse design.

It does:

1. read one sample
2. launch one `circuit-tracer attribute` subprocess
3. that subprocess loads model/transcoder
4. produces one `.pt`
5. exits
6. next sample starts fresh

### Why that mattered

That scheme can survive on tighter memory budgets because:

- it never tries to keep the full state alive for a long time
- per-sample failure does not poison the whole batch
- GPU memory gets released with process exit

This is why “old scripts could run transcoder” does **not** mean “single persistent answer-aligned process should also trivially fit”.

---

## Bug 12: `selected samples have no matched A/B .pt files`

### Symptom

The compare step failed with:

```text
ValueError: selected samples have no matched A/B .pt files
```

### Root cause

This error does **not** mean the compare script itself is wrong.

It means:

- no sample id had both:
  - `pt_a/<sample_id>.pt`
  - `pt_b/<sample_id>.pt`

Possible reasons:

- A succeeded, B failed
- B succeeded, A failed
- both sides succeeded, but for different sample_ids

### Correct debugging source

Always inspect:

- `answer_aligned_meta_a.csv`
- `answer_aligned_meta_b.csv`

These metadata files are the source of truth for sample-level failures.

Useful command:

```bash
python - <<'PY'
import csv

for p in [
    "outputs/phase_ab/ab_answer_aligned/ab_answer_smoke/answer_aligned_meta_a.csv",
    "outputs/phase_ab/ab_answer_aligned/ab_answer_smoke/answer_aligned_meta_b.csv",
]:
    print("\\n==", p, "==")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        print(
            r["sample_id"],
            "| status:", r["status"],
            "| target_token_id:", r.get("target_token_id", ""),
            "| error:", r["error_message"],
        )
PY
```

---

## Bug 13: sample-level `-9` exits

### Symptom

Metadata showed errors like:

```text
RuntimeError: attribute command failed with exit code -9
```

### Interpretation

This strongly suggests the sample subprocess was killed by the system, usually due to memory pressure.

### Fix attempts

We tried several strategies:

1. keep subprocess-per-sample but improve environment and stderr capture
2. try long-lived model reuse
3. later move to a single Python process with explicit cleanup

At this point, the robust lesson is:

- `-9` is not a normal Python exception
- it is a process-kill symptom
- focus on memory/load strategy, not just syntax

---

## Bug 14: batch process got fully `Killed`

### Symptom

The outer batch runner died with:

```text
Killed
```

before even finishing A.

### Root cause

When we tried a long-lived in-process reuse version, it initialized too much model + transcoder state at once. In addition:

- `ReplacementModel._configure_replacement_model(...)` currently does:

```python
transcoder_set.to(self.cfg.device, self.cfg.dtype)
```

So full transcoder state was being pushed to the GPU eagerly.

### Fix direction

We moved toward a more conservative “single Python process, sequential samples, explicit cleanup” strategy instead of one giant persistent in-memory session.

---

## Bug 15: lazy transcoder behavior mattered

### Symptom

A later OOM happened at startup before running samples.

### Root cause

An earlier refactor accidentally changed loading behavior to something effectively heavier than the old CLI path.

The old CLI path relied on lazy decoder behavior. We later restored that and then extended:

- [replacement_model.py](D:/code/Bridging/vlm-circuit-tracing/circuit_tracer_vlm/circuit_tracer/replacement_model.py)

to allow:

- `lazy_encoder`
- `lazy_decoder`

in `ReplacementModel.from_pretrained(...)`

Git commit:

- `15a8f44 enable lazy transcoders for sequential answer attribution`

---

## Current Answer-Aligned Strategy (latest stable direction)

The latest direction in:

- [run_batch_answer_aligned_attribute.py](D:/code/Bridging/vlm-circuit-tracing/circuit_tracer_vlm/scripts/research/run_batch_answer_aligned_attribute.py)

is:

- one Python process
- sequential sample loop
- per-sample:
  - derive assistant prefix
  - derive target answer token id
  - initialize `ReplacementModel` with lazy transcoders
  - run `attribute(...)`
  - save `.pt`
  - delete graph/model
  - run `gc.collect()`
  - run `torch.cuda.empty_cache()`

This is not “all samples share one persistent model instance”, because the current bottom-layer memory behavior is still too risky for that.

But it **is** single-process orchestration and is currently the most conservative in-repo path toward a stable answer-aligned run.

Git commits related to this chain:

- `2bea640 fix answer-aligned prompt image token`
- `1356e8c use current python for answer-aligned attribution`
- `eeb1d62 pin answer-aligned pipeline to venv python`
- `f312b16 prefer vendored transformer lens in package init`
- `cac10f2 match CLI lazy decoder behavior in answer attribution`
- `d18b6f7 stabilize answer attribution subprocess execution`
- `cbcb154 run answer attribution sequentially in one process`
- `15a8f44 enable lazy transcoders for sequential answer attribution`

---

## Useful Debug Commands

### Environment sanity

```bash
which python
which pip
python -c "import sys; print(sys.executable)"
python -m pip --version
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python -m pip --version
```

### Confirm venv deps

```bash
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python -c "import transformers, torch, einops, huggingface_hub; print('env ok')"
```

### Confirm vendored TransformerLens path

```bash
/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python - <<'PY'
from circuit_tracer import ReplacementModel
import transformer_lens
print("transformer_lens:", transformer_lens.__file__)
print("ReplacementModel import ok")
PY
```

### Confirm data extraction

```bash
find /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014 -name "*.jpg" | wc -l
```

Expected:

```bash
40504
```

### Inspect answer-aligned metadata

```bash
python - <<'PY'
import csv
for p in [
    "outputs/phase_ab/ab_answer_aligned/ab_answer_smoke/answer_aligned_meta_a.csv",
    "outputs/phase_ab/ab_answer_aligned/ab_answer_smoke/answer_aligned_meta_b.csv",
]:
    print("\\n==", p, "==")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        print(r["sample_id"], r["status"], r.get("used_max_feature_nodes",""), r["error_message"])
PY
```

### Inspect produced `.pt`

```bash
find outputs/phase_ab/ab_answer_aligned/ab_answer_smoke/pt_a -name "*.pt"
find outputs/phase_ab/ab_answer_aligned/ab_answer_smoke/pt_b -name "*.pt"
```

---

## Practical Lessons

1. Do not trust the shell prompt alone; always verify `which python`.
2. Do not assume old CSV paths match the new server.
3. For multimodal Gemma runs, prompt must carry `<start_of_image>` if images are passed directly.
4. `check_hf_access.py` proves connectivity, not full model readiness.
5. For compare-stage failures, always inspect `answer_aligned_meta_*.csv` first.
6. The old batch `.pt` workflow was subprocess-per-sample, not persistent-model reuse.
7. A real single-process answer-aligned pipeline is only safe if transcoder/device strategy is handled carefully.

---

## Recommended Operational Workflow Going Forward

### Local changes

Always change code locally in:

`D:\code\Bridging\vlm-circuit-tracing`

Then:

```bash
git add ...
git commit -m "..."
git push origin main
```

### Server sync

```bash
cd /root/autodl-tmp/tca-reasoning
git pull
cd circuit_tracer_vlm
source .venv/bin/activate
source scripts/server/load_env.sh .env
```

### Why

Do **not** hot-edit server code unless it is a true emergency.  
We lost too much time previously from:

- drifting local/server versions
- shell/environment confusion
- unclear provenance of fixes

Local-first + git is the correct habit here.

---

## Addendum: 2026-04-24 to 2026-04-25 Round

This addendum records the later debugging round after the earlier environment/path issues were already mostly solved. The focus here was:

- making answer-token-aligned tracing robust on the new 48 GB GPU server
- distinguishing true resource failures from pipeline logic bugs
- getting a stable overnight 4-bucket run at `MAX_FEATURE_NODES=1024`
- running controlled A/B compare successfully on all produced graphs

### A1. Initial symptom in this round: first sample got `Killed`

The first visible symptom was that:

```bash
bash scripts/server/run_ab_answer_aligned_trace_full.sh
```

would often die on the first sample with:

```text
Killed
```

The failure initially appeared near:

- HF checkpoint shard load completion
- `ReplacementModel` setup
- or early attribution setup

At first this looked like a generic OOM, but the exact stage was unclear.

### A2. Fix: add real stage-level logging and subprocess isolation

We improved observability in three places:

1. `scripts/research/run_batch_answer_aligned_attribute.py`
2. `circuit_tracer/attribution/attribute.py`
3. `circuit_tracer/replacement_model.py`

Main changes:

- subprocess-per-attempt execution mode for answer-aligned attribution
- per-attempt log files
- explicit metadata fields like `attempt_log_path`
- phase-level memory snapshots
- model-load logging before and after HF load / processor load / HookedVLTransformer construction

Why this mattered:

- before these changes, failures were opaque
- after these changes, we could tell whether the run died during:
  - HF model load
  - transcoder offload
  - forward pass
  - feature attribution
  - or compare

### A3. Bug: dtype was not truly passed into HF model load

Even though outer code passed `dtype=bfloat16`, the internal HF call in:

- `circuit_tracer/replacement_model.py`

was not correctly propagating that dtype into:

- `Gemma3ForConditionalGeneration.from_pretrained(...)`

This could increase the load-time memory peak.

Fix:

- explicitly pass `torch_dtype`
- enable `low_cpu_mem_usage=True`

Result:

- model-load logging became much more informative
- load behavior became closer to what we actually intended

### A4. Bug: `OFFLOAD=disk` failed because temp files went to the system disk

After switching from `OFFLOAD=cpu` to `OFFLOAD=disk`, we got a concrete error:

```text
safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)
```

Root cause:

- `disk_offload.py` used `tempfile.NamedTemporaryFile(...)`
- no explicit temp directory was provided
- so files were written under the system temp location on `/`
- the system disk was much smaller than the data disk

Operational fix:

```bash
mkdir -p /root/autodl-tmp/tmp
export TMPDIR=/root/autodl-tmp/tmp
```

Then always run answer-aligned attribution with:

```bash
OFFLOAD=disk
TMPDIR=/root/autodl-tmp/tmp
```

After that, disk offload became usable and was clearly better than trying to stay purely on GPU/CPU for this pipeline.

### A5. Bug: smoke tests accidentally ran hundreds of samples

At one point, a supposed small debug run was actually sampling:

- 4 buckets
- 60 samples per bucket

because the script defaults were being used.

This made the run seem inexplicably slow, but the issue was not mysterious performance; it was simply the wrong sampling scope.

Stable smoke-test pattern:

```bash
RUN_TAG=ab_one_pair \
BUCKETS=A0_B1 \
PER_BUCKET=1 \
RETRY_FEATURE_NODES="" \
STOP_ON_ATTR_ERROR=1 \
ANSWER_ATTR_VERBOSE_ATTRIBUTION=1 \
ANSWER_ATTR_EXEC_MODE=subprocess \
OFFLOAD=disk \
TMPDIR=/root/autodl-tmp/tmp \
bash scripts/server/run_ab_answer_aligned_trace_full.sh
```

### A6. Bug: `tokenization produced no answer token after prefix`

Once the resource path was stable enough to run a single sample, we hit:

```text
ValueError: tokenization produced no answer token after prefix
```

Root cause:

- the old logic assumed that tokenizing:
  - `assistant_prefix`
  - and `assistant_prefix + answer`
- would always let us recover the first answer token by simple length differencing

That assumption is not always true for Gemma tokenizer continuation behavior, especially when the answer tokenization depends on a leading space.

Fix:

- add fallback continuation-token logic in `_first_answer_token_id(...)`
- if the naive prefix-length differencing fails, tokenize the continuation more carefully

Result:

- answer-token alignment became robust enough to continue

### A7. Bug: compare initially failed because `.pt` graphs were never produced

An early compare error was:

```text
ValueError: selected samples have no matched A/B .pt files
```

Important clarification:

This did **not** mean compare was fundamentally wrong. It meant:

- A and/or B attribution had failed upstream
- no matched sample ids had both `pt_a` and `pt_b`

One important fix here was changing the batch runner so that if zero graphs were produced, it would stop instead of continuing into compare and emitting a confusing downstream error.

### A8. Bug: compare indexing/layout logic was wrong for new answer-aligned graphs

After we finally had `.pt` files, compare first failed with an index error and then with:

```text
ValueError: cannot infer error layout: n_errors=... is not divisible by n_pos=...
```

There were two separate causes.

#### A8.1 Compare-side layout inference was too brittle

`trace_compare_ab_controlled.py` used assumptions based too directly on:

- `len(input_tokens)`
- `cfg.n_layers`

That broke when the actual graph node layout did not match those assumptions exactly.

Fix:

- update `_build_index(...)`
- infer layout using the actual graph size from `adjacency_matrix.shape[0]`
- combine that with known feature/token/logit counts and layer structure

#### A8.2 Generation-side attribution did not actually use `assistant_prefix`

This was the more important bug.

Externally, the answer-aligned path constructed tokens for:

- `prompt + assistant_prefix`

But internally, `setup_attribution()` still did some processor / forward / cache work using only:

- `prompt`

That made the saved graph metadata internally inconsistent:

- `input_tokens` reflected one sequence
- the actual forward/layout reflected another

Fix:

- extend `ReplacementModel.setup_attribution(...)` to accept `assistant_prefix`
- build `full_prompt = prompt + assistant_prefix`
- use `full_prompt` consistently for:
  - processor calls
  - `run_with_hooks(...)`
  - cached batch/context state
- pass `assistant_prefix` from `attribute.py` into `setup_attribution(...)`

Result:

- newly generated `.pt` graphs became self-consistent
- compare could run successfully

### A9. Important lesson: "full graph" did not really mean full graph

We tested `MAX_FEATURE_NODES=0` with the intention of saving all active features.

Later, metadata inspection showed values like:

- `status = ok`
- `used_max_feature_nodes = 64`

This means:

- the true full-graph attempt failed
- the script retried with fallback feature budgets
- the saved graph was a truncated graph, not a real all-feature graph

So for this pipeline, the real source of truth is not just "run finished successfully"; it is:

- `answer_aligned_meta_a.csv`
- `answer_aligned_meta_b.csv`
- especially:
  - `used_max_feature_nodes`
  - `attempt_log_path`

### A10. Why true full graph is impractical here

The current graph representation stores a dense adjacency matrix.

That is manageable for:

- ~10k total nodes

but becomes unrealistic if active feature counts grow to:

- ~150k+

Because then the dense matrix would become enormous.

Conclusion:

- "save all active features" is not the right default target for this setup
- it is much more realistic to search for the largest stable capped graph budget

### A11. Stable budget found: `MAX_FEATURE_NODES=1024`

We tested a larger capped graph budget:

- `MAX_FEATURE_NODES=1024`

This was successful for:

- single-sample answer-aligned trace
- single-sample compare
- overnight bucket runs

Observed behavior:

- single-sample A/B compare ran successfully
- a bucket with 20 samples also ran successfully
- `.pt` file sizes were large but manageable

Approximate practical scale from this round:

- one sample A/B pair at 1024 features: about `~1 GB`
- one bucket of 20 samples: about `~20 GB`
- all 4 buckets x 20 samples: about `~82.7 GB`

This means:

- disk space on the data disk was sufficient
- the main bottleneck became runtime, not storage

### A12. Queue script introduced a small validator bug

We added a queue script:

- `scripts/server/run_ab_answer_aligned_trace_bucket_queue.sh`

and later validator / summary scripts:

- `scripts/research/validate_answer_aligned_queue.py`
- `scripts/research/show_answer_aligned_queue_results.py`

The overnight queue itself worked, but the first validator version falsely reported:

```text
bucket not found in queue status
```

even when all bucket outputs were present.

Root cause:

- `bucket_status.tsv` was tab-separated
- the validator read it like a comma-separated CSV

Fix:

- update `_read_csv(...)` in the validator so `.tsv` uses tab delimiter

This was a validator bug only, not a tracing/comparison failure.

### A13. Overnight result that is now known-good

The following overnight queue configuration completed successfully:

- 4 buckets
- 20 samples per bucket
- `MAX_FEATURE_NODES=1024`
- `SKIP_COMPARE=1` during the overnight trace stage

Then compare was run afterwards for each bucket and also completed successfully.

Final known-good result from this round:

- 4 buckets
- 80 samples total
- all A-side graphs produced
- all B-side graphs produced
- all 4 buckets successfully compared
- compare outputs include:
  - `sample_compare_controlled.csv`
  - `bucket_summary_controlled.csv`
  - `nodes_detailed_controlled.csv`
  - `edges_detailed_controlled.csv`

### A14. Current recommended stable operating mode

For future answer-aligned OK-VQA experiments on this server, the recommended default is:

```bash
OFFLOAD=disk
TMPDIR=/root/autodl-tmp/tmp
ANSWER_ATTR_EXEC_MODE=subprocess
MAX_FEATURE_NODES=1024
```

And for large overnight trace jobs:

```bash
SKIP_COMPARE=1
```

Then run compare afterwards as a separate stage.

This is currently the safest and most reproducible operational pattern we have verified.
