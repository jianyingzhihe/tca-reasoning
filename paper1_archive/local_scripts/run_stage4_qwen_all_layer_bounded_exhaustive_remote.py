#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")
BASE_RUNNER = ROOT / "scripts" / "local" / "run_stage4_qwen_plt_layer_sweep_remote.py"
PREFIX = "stage4_qwen_all_layer_bounded_exhaustive"


def _load_base():
    spec = importlib.util.spec_from_file_location("stage4_qwen_plt_layer_sweep_remote", BASE_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_cli(argv: list[str]) -> list[str]:
    out: list[str] = []
    has_layers = False
    has_top = False
    has_main = False
    has_max = False
    mode = "smoke"
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--mode" and idx + 1 < len(argv):
            mode = argv[idx + 1]
            out.extend([arg, mode])
            idx += 2
            continue
        if arg == "--layers" and idx + 1 < len(argv):
            has_layers = True
            layers = argv[idx + 1]
            if layers == "all":
                layers = ",".join(str(layer) for layer in range(28))
            out.extend([arg, layers])
            idx += 2
            continue
        if arg == "--top-per-prompt-run":
            has_top = True
        if arg == "--main-per-prompt-run":
            has_main = True
        if arg == "--max-prompt-runs":
            has_max = True
        out.append(arg)
        idx += 1
    if not has_layers:
        out.extend(["--layers", "0,7,14,21,27" if mode == "smoke" else ",".join(str(layer) for layer in range(28))])
    if not has_top:
        out.extend(["--top-per-prompt-run", "16"])
    if not has_main:
        out.extend(["--main-per-prompt-run", "8"])
    if not has_max and mode == "smoke":
        out.extend(["--max-prompt-runs", "6"])
    return out


def _pack_layer_block(base, pack: str, mode: str, layer: int, max_prompt_runs: int, top_per_prompt_run: int, main_per_prompt_run: int) -> str:
    runs_name = f"paperpack72_{pack}_prompt_runs.csv"
    cand = base._stem(pack, mode, layer, "candidates")
    zero = base._stem(pack, mode, layer, "zeroing")
    group = base._stem(pack, mode, layer, "group")
    max_candidates = 6 if mode == "smoke" else 0
    layer_main_cap = 256
    return f"""
echo '--- Stage4-044 all-layer bounded exhaustive {pack} L{layer} discovery ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_feature_discovery.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --prompt-runs "$STAGE/{runs_name}" \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{cand}.csv" \\
  --summary-json "$STAGE/{cand}.json" \\
  --layer {layer} \\
  --position-group visual_answer \\
  --candidate-pool-size 8192 \\
  --top-per-prompt-run {top_per_prompt_run} \\
  --main-per-prompt-run {main_per_prompt_run} \\
  --max-prompt-runs {max_prompt_runs}

echo '--- Stage4-044 cap main candidates to top {layer_main_cap} per layer ---'
.venv/bin/python - <<'PY'
import csv
from pathlib import Path
path = Path("$STAGE/{cand}.csv")
if path.exists() and path.stat().st_size:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    fields = list(rows[0].keys()) if rows else []
    main = [row for row in rows if row.get("include_main") == "1"]
    def score(row):
        try:
            return float(row.get("evidence_first_score") or 0)
        except ValueError:
            return 0.0
    keep = {{row.get("candidate_id", "") for row in sorted(main, key=score, reverse=True)[:{layer_main_cap}]}}
    for row in rows:
        if row.get("include_main") == "1" and row.get("candidate_id", "") not in keep:
            row["include_main"] = "0"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"stage4-044 candidate cap: rows={{len(rows)}} main_before={{len(main)}} main_after={{len(keep)}}")
PY

echo '--- Stage4-044 all-layer bounded exhaustive {pack} L{layer} zeroing controls ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{cand}.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --work-dir "$STAGE/work_{zero}" \\
  --out-csv "$STAGE/{zero}_raw.csv" \\
  --summary-json "$STAGE/{zero}_run.json" \\
  --layer {layer} \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-candidates {max_candidates}

echo '--- Stage4-044 all-layer bounded exhaustive {pack} L{layer} grouped restore ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{cand}.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{group}_raw.csv" \\
  --summary-json "$STAGE/{group}_run.json" \\
  --layer {layer} \\
  --top-ks 1,4,8,16,32,64,128 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-prompt-runs {max_prompt_runs}
"""


def main() -> int:
    base = _load_base()
    base.REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_all_layer_bounded_exhaustive"
    base.REMOTE_ASSETS = f"{base.REMOTE_STAGE}/assets"

    def stem(pack: str, mode: str, layer: int, artifact: str) -> str:
        return f"{PREFIX}_{pack}_{mode}_L{layer}_{artifact}"

    def status_command(mode: str, packs: list[str], layers: list[int]) -> str:
        patterns = "|".join(f"{PREFIX}_{pack}_{mode}_L{layer}" for pack in packs for layer in layers)
        return f"""
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'run_stage4_qwen_evidence_first|run_stage4_qwen_causal_cutter|{patterns}' | grep -v grep || true
echo FILES
ls -lh {base.REMOTE_STAGE}/{PREFIX}_*_{mode}_L* 2>/dev/null || true
echo LOGS
ls -lh {base.REMOTE_STAGE}/logs/* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""

    base._stem = stem
    base._status_command = status_command
    base._pack_layer_block = lambda pack, mode, layer, max_prompt_runs, top_per_prompt_run, main_per_prompt_run: _pack_layer_block(
        base, pack, mode, layer, max_prompt_runs, top_per_prompt_run, main_per_prompt_run
    )
    sys.argv = [str(BASE_RUNNER), *_parse_cli(sys.argv[1:]), "--skip-analyze"]
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
