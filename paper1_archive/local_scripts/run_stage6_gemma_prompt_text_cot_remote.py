#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import posixpath
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage6_gemma_prompt_text_cot"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage6_gemma_prompt_text_cot"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner",
        ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_text(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def _put_text(base, sftp, remote_path: str, text: str) -> None:
    base._mkdir_p(sftp, posixpath.dirname(remote_path))
    with sftp.file(remote_path, "w") as handle:
        handle.write(text.replace("\r\n", "\n"))


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _manifest_paths(tag: str) -> tuple[Path, Path, Path]:
    return (
        LOCAL_CROSS / f"{PREFIX}_{tag}_eval_A_condition.csv",
        LOCAL_CROSS / f"{PREFIX}_{tag}_eval_B_baseline.csv",
        LOCAL_CROSS / f"{PREFIX}_{tag}_manifest.csv",
    )


def _filter(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    key = "stage6_include_smoke" if mode == "smoke" else "stage6_include_full"
    return [row for row in rows if row.get(key, "1") == "1"]


def _remote_image_path(image_filename: str) -> str:
    return f"{REMOTE_ASSETS}/images/{Path(image_filename).name}"


def _remoteize_eval_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Path]]:
    out: list[dict[str, str]] = []
    images: dict[str, Path] = {}
    for row in rows:
        item = dict(row)
        image_path = Path(item.get("image_path", ""))
        image_name = Path(item.get("image_filename", "")).name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        item["image_path"] = _remote_image_path(image_name)
        images[image_name] = image_path
        out.append(item)
    return out, images


def _upload_pack(base, sftp, tag: str, mode: str) -> dict[str, int]:
    a_path, b_path, manifest_path = _manifest_paths(tag)
    if not a_path.exists() or not b_path.exists() or not manifest_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "build_stage6_gemma_prompt_text_pack.py"),
                "--tag",
                tag,
            ],
            check=True,
        )
    a_rows = _filter(_read_csv(a_path), mode)
    b_rows = _filter(_read_csv(b_path), mode)
    manifest_rows = _filter(_read_csv(manifest_path), mode)
    if not a_rows or not b_rows or not manifest_rows:
        raise ValueError(f"empty Stage6 Gemma {mode} pack: A={len(a_rows)} B={len(b_rows)} manifest={len(manifest_rows)}")

    a_remote, a_images = _remoteize_eval_rows(a_rows)
    b_remote, b_images = _remoteize_eval_rows(b_rows)
    images = {**a_images, **b_images}
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    for image_name, image_path in sorted(images.items()):
        base._put_file(sftp, image_path, f"{REMOTE_ASSETS}/images/{image_name}")

    remote_a = f"{REMOTE_STAGE}/{PREFIX}_{mode}_{tag}_eval_A_condition.csv"
    remote_b = f"{REMOTE_STAGE}/{PREFIX}_{mode}_{tag}_eval_B_baseline.csv"
    remote_manifest = f"{REMOTE_STAGE}/{PREFIX}_{mode}_{tag}_manifest.csv"
    _put_text(base, sftp, remote_a, _csv_text(a_remote, list(a_remote[0].keys())))
    _put_text(base, sftp, remote_b, _csv_text(b_remote, list(b_remote[0].keys())))
    _put_text(base, sftp, remote_manifest, _csv_text(manifest_rows, list(manifest_rows[0].keys())))
    print(
        f"uploaded Stage6 Gemma pack mode={mode}: A={len(a_remote)} B={len(b_remote)} "
        f"manifest={len(manifest_rows)} images={len(images)}",
        flush=True,
    )
    return {"a_rows": len(a_remote), "b_rows": len(b_remote), "manifest_rows": len(manifest_rows), "images": len(images)}


def _retry_feature_nodes(max_feature_nodes: int) -> str:
    candidates = [48, 32, 24, 16, 8]
    values = [value for value in candidates if value < max_feature_nodes]
    return ",".join(str(value) for value in values[:3]) or "16,8"


def _min_free_gb(mode: str, max_feature_nodes: int, allow_large_graphs: bool, sharded_streaming: bool = False) -> int:
    if mode == "smoke":
        return 20
    if sharded_streaming:
        if max_feature_nodes <= 8:
            return 35
        if not allow_large_graphs:
            raise ValueError(
                "Stage6 Gemma sharded full with max_feature_nodes > 8 is blocked by default. "
                "Use --allow-large-graphs only after confirming enough disk headroom."
            )
        return 80
    if max_feature_nodes <= 8:
        return 80
    if not allow_large_graphs:
        raise ValueError(
            "Stage6 Gemma full with max_feature_nodes > 8 is blocked by default. "
            "Use --allow-large-graphs only after confirming enough disk headroom."
        )
    return 160


def _assert_local_cleanup_evidence(mode: str, tag: str) -> None:
    prefix = f"{PREFIX}_{mode}_{tag}"
    required = [
        LOCAL_CROSS / f"{prefix}_decision.json",
        LOCAL_CROSS / f"{prefix}_eval_A_condition.csv",
        LOCAL_CROSS / f"{prefix}_eval_B_baseline.csv",
        LOCAL_CROSS / f"{prefix}_valid_samples.csv",
        LOCAL_CROSS / f"{prefix}_failure_manifest.csv",
        LOCAL_CROSS / f"{prefix}_meta_a.csv",
        LOCAL_CROSS / f"{prefix}_meta_b.csv",
    ]
    missing = [path for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Refusing remote cleanup until local CSV/JSON evidence is present: "
            + ", ".join(str(path) for path in missing)
        )


def _cleanup_failed_run_command(mode: str, tag: str) -> str:
    run_root = f"{REMOTE_STAGE}/source_tracing_{mode}_{tag}"
    return f"""
set -euo pipefail
STAGE={REMOTE_STAGE}
RUN_ROOT={run_root}
case "$RUN_ROOT" in
  "$STAGE"/source_tracing_*) ;;
  *) echo "unsafe RUN_ROOT=$RUN_ROOT"; exit 99 ;;
esac
echo CLEANUP_TARGET "$RUN_ROOT"
echo PROCS
if ps -eo pid,ppid,stat,etime,args | grep -F "$RUN_ROOT" | grep -v grep; then
  echo "refusing cleanup: process still references RUN_ROOT=$RUN_ROOT"
  exit 88
fi
echo BEFORE_DF
df -h /root/autodl-tmp
echo BEFORE_DU
du -sh "$RUN_ROOT" "$RUN_ROOT/graphs_a_condition" "$RUN_ROOT/graphs_b_baseline" "$RUN_ROOT/graphs_b_baseline_unique" "$RUN_ROOT/compare" 2>/dev/null || true
for target in "$RUN_ROOT/graphs_a_condition" "$RUN_ROOT/graphs_b_baseline" "$RUN_ROOT/graphs_b_baseline_unique" "$RUN_ROOT/compare"; do
  case "$target" in
    "$STAGE"/source_tracing_*/*) rm -rf "$target" ;;
    *) echo "unsafe cleanup target=$target"; exit 99 ;;
  esac
done
echo AFTER_DF
df -h /root/autodl-tmp
echo AFTER_DU
du -sh "$RUN_ROOT" 2>/dev/null || true
"""


def _remote_sharded_script(
    mode: str,
    tag: str,
    resume: bool,
    max_new_tokens: int,
    max_feature_nodes: int,
    min_free_gb: int,
    shard_count: int,
    cleanup_graphs_after_compare: bool,
) -> str:
    run_root = f"{REMOTE_STAGE}/source_tracing_{mode}_{tag}"
    shard_root = f"{run_root}/shards"
    compare_dir = f"{run_root}/compare"
    bucket = f"stage6_gemma_prompt_text_cot_{tag}"
    cleanup = (
        f"""case "$RUN_ROOT" in
  "$STAGE"/source_tracing_*) rm -rf "$RUN_ROOT" ;;
  *) echo "unsafe RUN_ROOT=$RUN_ROOT"; exit 99 ;;
esac
"""
        if not resume
        else 'echo "[resume] preserving RUN_ROOT=$RUN_ROOT"\n'
    )
    eval_resume_flag = "--resume" if resume else "--no-resume"
    retry_feature_nodes = _retry_feature_nodes(max_feature_nodes)
    cleanup_flag = "1" if cleanup_graphs_after_compare else "0"
    return f"""#!/usr/bin/env bash
set -uo pipefail
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub

STAGE={REMOTE_STAGE}
RUN_ROOT={run_root}
SHARD_ROOT={shard_root}
MERGED_COMPARE_DIR={compare_dir}
BUCKET={bucket}
MIN_FREE_GB={min_free_gb}
SHARD_COUNT={shard_count}
CLEANUP_GRAPHS_AFTER_COMPARE={cleanup_flag}
MANIFEST_A_FULL="$STAGE/{PREFIX}_{mode}_{tag}_eval_A_condition.csv"
MANIFEST_B_FULL="$STAGE/{PREFIX}_{mode}_{tag}_eval_B_baseline.csv"
BUCKET_CSV_FULL="$STAGE/{PREFIX}_{mode}_{tag}_manifest.csv"
DECISION_JSON="$STAGE/{PREFIX}_{mode}_{tag}_decision.json"

write_decision() {{
  local status="$1"
  local failure_type="$2"
  local failed_step="$3"
  .venv/bin/python - "$status" "$failure_type" "$failed_step" <<'PY'
import csv, json, sys
from pathlib import Path
status, failure_type, failed_step = sys.argv[1:4]
stage = Path("{REMOTE_STAGE}")
run_root = Path("{run_root}")
shard_root = Path("{shard_root}")
compare_dir = Path("{compare_dir}")
prefix = "{PREFIX}_{mode}_{tag}"
def count_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))
def count_graphs(relative):
    total = 0
    for shard_dir in shard_root.glob("shard_*"):
        graph_dir = shard_dir / relative
        if graph_dir.exists():
            total += len(list(graph_dir.glob("*.pt")))
    return total
payload = {{
    "status": status,
    "failure_type": failure_type,
    "failed_step": failed_step,
    "mode": "{mode}",
    "tag": "{tag}",
    "bucket": "{bucket}",
    "sharded_streaming": True,
    "shard_count_requested": {shard_count},
    "cleanup_graphs_after_compare": {cleanup_graphs_after_compare!r},
    "interpretation": "Gemma Stage6 prompt/text/CoT source-tracing counterpart, sharded to avoid persistent .pt graph growth.",
    "counts": {{
        "eval_a_rows": count_csv(run_root / "promptA_condition_eval.csv"),
        "eval_b_rows": count_csv(run_root / "promptB_baseline_eval.csv"),
        "valid_sample_rows": count_csv(run_root / f"{{prefix}}_valid_samples.csv"),
        "failure_rows": count_csv(run_root / f"{{prefix}}_failure_manifest.csv"),
        "meta_a_rows": count_csv(run_root / "answer_aligned_meta_a.csv"),
        "meta_b_rows": count_csv(run_root / "answer_aligned_meta_b.csv"),
        "sample_compare_rows": count_csv(compare_dir / "sample_compare_controlled.csv"),
        "nodes_detailed_rows": count_csv(compare_dir / "nodes_detailed_controlled.csv"),
        "edges_detailed_rows": count_csv(compare_dir / "edges_detailed_controlled.csv"),
        "shard_index_rows": count_csv(run_root / "shard_index.csv"),
        "graph_a_files_after_cleanup": count_graphs("graphs_a_condition"),
        "graph_b_files_after_cleanup": count_graphs("graphs_b_baseline"),
        "graph_b_unique_files_after_cleanup": count_graphs("graphs_b_baseline_unique"),
    }},
    "artifacts": {{
        "shard_index": str(run_root / "shard_index.csv"),
        "eval_a": str(run_root / "promptA_condition_eval.csv"),
        "eval_b": str(run_root / "promptB_baseline_eval.csv"),
        "valid_samples": str(run_root / f"{{prefix}}_valid_samples.csv"),
        "failure_manifest": str(run_root / f"{{prefix}}_failure_manifest.csv"),
        "meta_a": str(run_root / "answer_aligned_meta_a.csv"),
        "meta_b": str(run_root / "answer_aligned_meta_b.csv"),
        "compare_sample": str(compare_dir / "sample_compare_controlled.csv"),
        "compare_nodes": str(compare_dir / "nodes_detailed_controlled.csv"),
        "compare_edges": str(compare_dir / "edges_detailed_controlled.csv"),
    }},
}}
(stage / f"{{prefix}}_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}}

disk_free_gb() {{
  df -BG --output=avail /root/autodl-tmp | tail -n 1 | tr -dc '0-9'
}}

log_disk() {{
  local label="$1"
  echo "--- disk:$label ---"
  df -h /root/autodl-tmp
  du -sh "$RUN_ROOT" "$SHARD_ROOT" "$MERGED_COMPARE_DIR" 2>/dev/null || true
}}

require_free_gb() {{
  local label="$1"
  local required="$2"
  local free_gb
  free_gb=$(disk_free_gb)
  echo "[disk_gate] label=$label free_gb=$free_gb required_gb=$required"
  if [ -z "$free_gb" ] || [ "$free_gb" -lt "$required" ]; then
    write_decision "blocked" "disk_free_below_threshold" "$label"
    exit 75
  fi
}}

run_step() {{
  local name="$1"
  local failure_type="$2"
  shift 2
  echo "--- $name ---"
  log_disk "$name:before"
  "$@"
  local code=$?
  log_disk "$name:after"
  if [ "$code" -ne 0 ]; then
    write_decision "blocked" "$failure_type" "$name"
    exit "$code"
  fi
}}

run_attr_step() {{
  local name="$1"
  local graph_dir="$2"
  local min_pt_count="$3"
  shift 3
  echo "--- $name ---"
  log_disk "$name:before"
  "$@"
  local code=$?
  log_disk "$name:after"
  local pt_count=0
  if [ -d "$graph_dir" ]; then
    pt_count=$(find "$graph_dir" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')
  fi
  echo "[attr] $name exit=$code graph_pt_count=$pt_count min_pt_count=$min_pt_count"
  if [ "$pt_count" -lt "$min_pt_count" ]; then
    write_decision "blocked" "answer_aligned_attribute_oom_or_graph_missing" "$name"
    if [ "$code" -eq 0 ]; then exit 75; else exit "$code"; fi
  fi
  if [ "$code" -ne 0 ]; then
    echo "[warn] $name had per-sample attribution failures; continuing because graph_pt_count=$pt_count"
  fi
}}

safe_cleanup_graphs() {{
  local shard_dir="$1"
  case "$shard_dir" in
    "$RUN_ROOT"/shards/shard_*) ;;
    *) echo "unsafe shard_dir=$shard_dir"; exit 99 ;;
  esac
  echo "--- cleanup_graphs_after_compare $(basename "$shard_dir") ---"
  du -sh "$shard_dir/graphs_a_condition" "$shard_dir/graphs_b_baseline_unique" "$shard_dir/graphs_b_baseline" 2>/dev/null || true
  rm -rf "$shard_dir/graphs_a_condition" "$shard_dir/graphs_b_baseline_unique" "$shard_dir/graphs_b_baseline"
  log_disk "after_graph_cleanup:$(basename "$shard_dir")"
}}

echo '--- Stage6 Gemma prompt/text/CoT sharded source-tracing disk/gpu ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo "mode={mode} tag={tag} resume={str(resume).lower()} max_new_tokens={max_new_tokens} max_feature_nodes={max_feature_nodes} sharded_streaming=true shard_count=$SHARD_COUNT cleanup_graphs_after_compare=$CLEANUP_GRAPHS_AFTER_COMPARE"
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

{cleanup}
mkdir -p "$RUN_ROOT" "$SHARD_ROOT" "$MERGED_COMPARE_DIR"
log_disk "after_cleanup_before_preflight"
require_free_gb "preflight_sharded" "$MIN_FREE_GB"

run_step "py_compile_gemma_stage6_research_scripts" "local_or_remote_script_compile" \\
  .venv/bin/python -m py_compile \\
  scripts/research/run_batch_eval.py \\
  scripts/research/run_batch_answer_aligned_attribute.py \\
  scripts/research/trace_compare_ab_controlled.py

run_step "build_shard_manifests" "shard_manifest_build" \\
  .venv/bin/python - "$MANIFEST_A_FULL" "$MANIFEST_B_FULL" "$BUCKET_CSV_FULL" "$SHARD_ROOT" "$RUN_ROOT/shard_index.csv" "$SHARD_COUNT" <<'PY'
import csv, sys
from pathlib import Path
manifest_a, manifest_b, bucket_csv, shard_root, shard_index, shard_count = sys.argv[1:7]
manifest_a = Path(manifest_a)
manifest_b = Path(manifest_b)
bucket_csv = Path(bucket_csv)
shard_root = Path(shard_root)
shard_index = Path(shard_index)
shard_count = int(shard_count)
def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
def write_rows(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["sample_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({{key: row.get(key, "") for key in fieldnames}})
a_rows = read_rows(manifest_a)
b_rows = read_rows(manifest_b)
bucket_rows = read_rows(bucket_csv)
if not bucket_rows:
    raise SystemExit("empty bucket manifest")
originals = sorted({{row.get("stage6_original_sample_id") or row["sample_id"].split("__", 1)[0] for row in bucket_rows}})
if not originals:
    raise SystemExit("no original samples to shard")
effective_shards = max(1, min(shard_count, len(originals)))
original_to_shard = {{sample_id: idx % effective_shards for idx, sample_id in enumerate(originals)}}
sample_to_shard = {{}}
for row in bucket_rows:
    original = row.get("stage6_original_sample_id") or row["sample_id"].split("__", 1)[0]
    sample_to_shard[row["sample_id"]] = original_to_shard[original]
summary_rows = []
for shard_idx in range(effective_shards):
    shard_name = f"shard_{{shard_idx:02d}}"
    shard_dir = shard_root / shard_name
    shard_samples = {{sample_id for sample_id, sid in sample_to_shard.items() if sid == shard_idx}}
    shard_originals = sorted({{row.get("stage6_original_sample_id") or row["sample_id"].split("__", 1)[0] for row in bucket_rows if row["sample_id"] in shard_samples}})
    shard_bucket = [row for row in bucket_rows if row["sample_id"] in shard_samples]
    shard_a = [row for row in a_rows if row.get("sample_id") in shard_samples]
    shard_b = [row for row in b_rows if row.get("sample_id") in shard_samples]
    write_rows(shard_dir / "input_A_condition.csv", shard_a, list(a_rows[0].keys()) if a_rows else None)
    write_rows(shard_dir / "input_B_baseline.csv", shard_b, list(b_rows[0].keys()) if b_rows else None)
    write_rows(shard_dir / "bucket_manifest.csv", shard_bucket, list(bucket_rows[0].keys()))
    summary_rows.append({{
        "shard_id": shard_name,
        "original_sample_count": str(len(shard_originals)),
        "condition_count": str(len(shard_bucket)),
        "eval_a_rows": str(len(shard_a)),
        "eval_b_rows": str(len(shard_b)),
        "original_samples": ";".join(shard_originals),
    }})
write_rows(shard_index, summary_rows, ["shard_id", "original_sample_count", "condition_count", "eval_a_rows", "eval_b_rows", "original_samples"])
print(f"[shards] originals={{len(originals)}} effective_shards={{effective_shards}} index={{shard_index}}")
for row in summary_rows:
    print("[shard]", row)
PY

for SHARD_DIR in "$SHARD_ROOT"/shard_*; do
  if [ ! -d "$SHARD_DIR" ]; then
    continue
  fi
  SHARD_ID=$(basename "$SHARD_DIR")
  A_INPUT="$SHARD_DIR/input_A_condition.csv"
  B_INPUT="$SHARD_DIR/input_B_baseline.csv"
  BUCKET_CSV="$SHARD_DIR/bucket_manifest.csv"
  VALID_BUCKET_CSV="$SHARD_DIR/{PREFIX}_{mode}_{tag}_${{SHARD_ID}}_valid_samples.csv"
  FAILURE_CSV="$SHARD_DIR/{PREFIX}_{mode}_{tag}_${{SHARD_ID}}_failure_manifest.csv"
  EVAL_A="$SHARD_DIR/promptA_condition_eval.csv"
  EVAL_B="$SHARD_DIR/promptB_baseline_eval.csv"
  GRAPH_A="$SHARD_DIR/graphs_a_condition"
  GRAPH_B="$SHARD_DIR/graphs_b_baseline"
  GRAPH_B_UNIQUE="$SHARD_DIR/graphs_b_baseline_unique"
  META_A="$SHARD_DIR/answer_aligned_meta_a.csv"
  META_B="$SHARD_DIR/answer_aligned_meta_b.csv"
  B_UNIQUE_EVAL="$SHARD_DIR/promptB_baseline_unique_eval.csv"
  B_UNIQUE_SELECTED="$SHARD_DIR/baseline_unique_selected.csv"
  B_COPY_MAP="$SHARD_DIR/baseline_copy_map.csv"
  COMPARE_DIR="$SHARD_DIR/compare"
  mkdir -p "$COMPARE_DIR"
  echo "=== shard_start $SHARD_ID ==="
  log_disk "$SHARD_ID:start"
  require_free_gb "$SHARD_ID:pre_attr" "$MIN_FREE_GB"

  run_step "$SHARD_ID:eval_A_condition" "eval_prompt_or_gemma_model_load" \\
    .venv/bin/python -u scripts/research/run_batch_eval.py \\
    --manifest "$A_INPUT" \\
    --output-csv "$EVAL_A" \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --max-new-tokens {max_new_tokens} \\
    --correct-rule strict_gold \\
    {eval_resume_flag} \\
    --log-every 5

  run_step "$SHARD_ID:eval_B_baseline" "eval_prompt_or_gemma_model_load" \\
    .venv/bin/python -u scripts/research/run_batch_eval.py \\
    --manifest "$B_INPUT" \\
    --output-csv "$EVAL_B" \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --max-new-tokens {max_new_tokens} \\
    --correct-rule strict_gold \\
    {eval_resume_flag} \\
    --log-every 5

  run_step "$SHARD_ID:build_valid_sample_csv" "eval_empty_generation_or_target_alignment" \\
    .venv/bin/python - "$EVAL_A" "$EVAL_B" "$BUCKET_CSV" "$VALID_BUCKET_CSV" "$FAILURE_CSV" <<'PY'
import csv, sys
from pathlib import Path
eval_a, eval_b, bucket_csv, out_csv, failure_csv = [Path(arg) for arg in sys.argv[1:6]]
def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
def usable(row):
    return bool((row.get("generated_text") or "").strip()) and not (row.get("error_message") or "").strip()
def reason(row, side):
    if not row:
        return f"missing_eval_{{side}}"
    if (row.get("error_message") or "").strip():
        return f"error_{{side}}:" + (row.get("error_message") or "").strip()
    if not (row.get("generated_text") or "").strip():
        return f"empty_generated_{{side}}"
    return ""
a = {{row["sample_id"]: row for row in read_rows(eval_a)}}
b = {{row["sample_id"]: row for row in read_rows(eval_b)}}
bucket_rows = read_rows(bucket_csv)
valid, failures = [], []
for row in bucket_rows:
    sid = row["sample_id"]
    a_row = a.get(sid, {{}})
    b_row = b.get(sid, {{}})
    a_ok = usable(a_row)
    b_ok = usable(b_row)
    if a_ok and b_ok:
        valid.append(row)
    else:
        failures.append({{
            "sample_id": sid,
            "bucket": row.get("bucket", ""),
            "stage6_original_sample_id": row.get("stage6_original_sample_id", ""),
            "stage6_prompt_family": row.get("stage6_prompt_family", ""),
            "stage6_question_variant": row.get("stage6_question_variant", ""),
            "answer_text": row.get("answer_text", ""),
            "usable_a": "1" if a_ok else "0",
            "usable_b": "1" if b_ok else "0",
            "failure_a": reason(a_row, "a"),
            "failure_b": reason(b_row, "b"),
            "generated_a": a_row.get("generated_text", ""),
            "generated_b": b_row.get("generated_text", ""),
        }})
out_csv.parent.mkdir(parents=True, exist_ok=True)
valid_fields = list(bucket_rows[0].keys()) if bucket_rows else ["sample_id", "bucket"]
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=valid_fields)
    writer.writeheader()
    for row in valid:
        writer.writerow({{key: row.get(key, "") for key in valid_fields}})
failure_fields = [
    "sample_id", "bucket", "stage6_original_sample_id", "stage6_prompt_family",
    "stage6_question_variant", "answer_text", "usable_a", "usable_b",
    "failure_a", "failure_b", "generated_a", "generated_b",
]
with failure_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=failure_fields)
    writer.writeheader()
    writer.writerows(failures)
print(f"[valid] nonempty in both prompts={{len(valid)}}/{{len(bucket_rows)}}")
print(f"[failures] rows={{len(failures)}} -> {{failure_csv}}")
if len(valid) < 1:
    raise SystemExit("fewer than 1 valid sample after eval filtering")
PY

  run_attr_step "$SHARD_ID:attribute_A_condition_gold_answer" "$GRAPH_A" 1 \\
    .venv/bin/python -u scripts/research/run_batch_answer_aligned_attribute.py \\
    --eval-csv "$EVAL_A" \\
    --output-dir "$GRAPH_A" \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --selected-csv "$VALID_BUCKET_CSV" \\
    --answer-source gold \\
    --metadata-csv "$META_A" \\
    --max-feature-nodes {max_feature_nodes} \\
    --retry-feature-nodes {retry_feature_nodes} \\
    --exec-mode subprocess

  run_step "$SHARD_ID:build_unique_B_baseline_graph_queue" "baseline_dedup_queue_build" \\
    .venv/bin/python - "$EVAL_B" "$VALID_BUCKET_CSV" "$B_UNIQUE_EVAL" "$B_UNIQUE_SELECTED" "$B_COPY_MAP" <<'PY'
import csv, sys
from pathlib import Path
eval_b, valid_csv, unique_eval, unique_selected, copy_map = [Path(arg) for arg in sys.argv[1:6]]
def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
eval_rows = {{row["sample_id"]: row for row in read_rows(eval_b)}}
valid_rows = read_rows(valid_csv)
first_by_original = {{}}
copy_rows = []
for row in valid_rows:
    sid = row["sample_id"]
    original = row.get("stage6_original_sample_id") or sid.split("__", 1)[0]
    first_by_original.setdefault(original, sid)
for row in valid_rows:
    sid = row["sample_id"]
    original = row.get("stage6_original_sample_id") or sid.split("__", 1)[0]
    copy_rows.append({{"source_sample_id": first_by_original[original], "target_sample_id": sid, "stage6_original_sample_id": original}})
selected_ids = set(first_by_original.values())
unique_eval_rows = [eval_rows[sid] for sid in selected_ids if sid in eval_rows]
unique_selected_rows = [row for row in valid_rows if row["sample_id"] in selected_ids]
for path, rows in [(unique_eval, unique_eval_rows), (unique_selected, unique_selected_rows), (copy_map, copy_rows)]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        continue
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[baseline_dedup] unique_originals={{len(selected_ids)}} valid_conditions={{len(valid_rows)}} unique_eval_rows={{len(unique_eval_rows)}}")
if len(unique_eval_rows) < 1:
    raise SystemExit("fewer than 1 unique baseline row")
PY

  run_attr_step "$SHARD_ID:attribute_B_baseline_gold_answer_unique" "$GRAPH_B_UNIQUE" 1 \\
    .venv/bin/python -u scripts/research/run_batch_answer_aligned_attribute.py \\
    --eval-csv "$B_UNIQUE_EVAL" \\
    --output-dir "$GRAPH_B_UNIQUE" \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --selected-csv "$B_UNIQUE_SELECTED" \\
    --answer-source gold \\
    --metadata-csv "$META_B" \\
    --max-feature-nodes {max_feature_nodes} \\
    --retry-feature-nodes {retry_feature_nodes} \\
    --exec-mode subprocess

  run_step "$SHARD_ID:copy_unique_B_baseline_graphs_to_conditions" "baseline_graph_copy_missing" \\
    .venv/bin/python - "$GRAPH_B_UNIQUE" "$GRAPH_B" "$B_COPY_MAP" <<'PY'
import csv, os, sys
from pathlib import Path
src_dir, dst_dir, copy_map = [Path(arg) for arg in sys.argv[1:4]]
dst_dir.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(copy_map.open("r", encoding="utf-8", newline="")))
copied = 0
linked = 0
missing = []
link_errors = []
for row in rows:
    src = src_dir / f"{{row['source_sample_id']}}.pt"
    dst = dst_dir / f"{{row['target_sample_id']}}.pt"
    if not src.exists():
        missing.append(str(src))
        continue
    if src.resolve() != dst.resolve():
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            try:
                dst.symlink_to(src)
                linked += 1
            except OSError as exc:
                link_errors.append(f"{{src}} -> {{dst}}: {{exc}}")
    copied += 1
print(f"[baseline_dedup] materialized={{copied}} linked_or_symlinked={{linked}} missing={{len(missing)}} link_errors={{len(link_errors)}} dst={{dst_dir}}")
if missing:
    print("\\n".join(missing[:20]))
    raise SystemExit("missing unique baseline graph(s)")
if link_errors:
    print("\\n".join(link_errors[:20]))
    raise SystemExit("hardlink/symlink failed; refusing to copy large graph files")
PY

  run_step "$SHARD_ID:trace_compare_ab_controlled" "compare_format_or_graph_missing" \\
    .venv/bin/python -u scripts/research/trace_compare_ab_controlled.py \\
    --pt-dir-a "$GRAPH_A" \\
    --pt-dir-b "$GRAPH_B" \\
    --bucket-csv "$VALID_BUCKET_CSV" \\
    --out-dir "$COMPARE_DIR" \\
    --buckets "$BUCKET" \\
    --per-bucket 999 \\
    --topk-per-node 16 \\
    --beam-per-depth 16 \\
    --coverage 0.85 \\
    --max-depth 4 \\
    --min-abs-weight 0

  if [ "$CLEANUP_GRAPHS_AFTER_COMPARE" = "1" ]; then
    safe_cleanup_graphs "$SHARD_DIR"
  fi
  echo "=== shard_done $SHARD_ID ==="
done

run_step "merge_shard_outputs" "shard_output_merge" \\
  .venv/bin/python - "$RUN_ROOT" "$SHARD_ROOT" "$MERGED_COMPARE_DIR" "{PREFIX}_{mode}_{tag}" <<'PY'
import csv, sys
from pathlib import Path
run_root, shard_root, merged_compare, prefix = sys.argv[1:5]
run_root = Path(run_root)
shard_root = Path(shard_root)
merged_compare = Path(merged_compare)
def read_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({{key: row.get(key, "") for key in fields}})
def concat(relative, output):
    rows = []
    for shard_dir in sorted(shard_root.glob("shard_*")):
        for row in read_rows(shard_dir / relative):
            row = dict(row)
            row.setdefault("stage6_shard_id", shard_dir.name)
            rows.append(row)
    write_rows(output, rows)
    print("[merge]", relative, "->", output, "rows", len(rows))
concat("promptA_condition_eval.csv", run_root / "promptA_condition_eval.csv")
concat("promptB_baseline_eval.csv", run_root / "promptB_baseline_eval.csv")
concat("answer_aligned_meta_a.csv", run_root / "answer_aligned_meta_a.csv")
concat("answer_aligned_meta_b.csv", run_root / "answer_aligned_meta_b.csv")
concat("compare/sample_compare_controlled.csv", merged_compare / "sample_compare_controlled.csv")
concat("compare/bucket_summary_controlled.csv", merged_compare / "bucket_summary_controlled.csv")
concat("compare/nodes_detailed_controlled.csv", merged_compare / "nodes_detailed_controlled.csv")
concat("compare/edges_detailed_controlled.csv", merged_compare / "edges_detailed_controlled.csv")
valid_rows = []
failure_rows = []
for shard_dir in sorted(shard_root.glob("shard_*")):
    valid_rows.extend(read_rows(shard_dir / f"{{prefix}}_{{shard_dir.name}}_valid_samples.csv"))
    failure_rows.extend(read_rows(shard_dir / f"{{prefix}}_{{shard_dir.name}}_failure_manifest.csv"))
write_rows(run_root / f"{{prefix}}_valid_samples.csv", valid_rows)
write_rows(run_root / f"{{prefix}}_failure_manifest.csv", failure_rows)
if not read_rows(merged_compare / "sample_compare_controlled.csv"):
    raise SystemExit("merged sample_compare_controlled.csv is empty")
PY

write_decision "pass" "" ""
log_disk "final_after_merge"
"""


def _remote_script(
    mode: str,
    tag: str,
    resume: bool,
    max_new_tokens: int,
    max_feature_nodes: int,
    min_free_gb: int,
) -> str:
    run_root = f"{REMOTE_STAGE}/source_tracing_{mode}_{tag}"
    compare_dir = f"{run_root}/compare"
    bucket = f"stage6_gemma_prompt_text_cot_{tag}"
    cleanup = (
        f"""case "$RUN_ROOT" in
  "$STAGE"/source_tracing_*) rm -rf "$RUN_ROOT" ;;
  *) echo "unsafe RUN_ROOT=$RUN_ROOT"; exit 99 ;;
esac
"""
        if not resume
        else 'echo "[resume] preserving RUN_ROOT=$RUN_ROOT"\n'
    )
    eval_resume_flag = "--resume" if resume else "--no-resume"
    retry_feature_nodes = _retry_feature_nodes(max_feature_nodes)
    return f"""#!/usr/bin/env bash
set -uo pipefail
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub

STAGE={REMOTE_STAGE}
RUN_ROOT={run_root}
COMPARE_DIR={compare_dir}
BUCKET={bucket}
MIN_FREE_GB={min_free_gb}
MANIFEST_A="$STAGE/{PREFIX}_{mode}_{tag}_eval_A_condition.csv"
MANIFEST_B="$STAGE/{PREFIX}_{mode}_{tag}_eval_B_baseline.csv"
BUCKET_CSV="$STAGE/{PREFIX}_{mode}_{tag}_manifest.csv"
VALID_BUCKET_CSV="$RUN_ROOT/{PREFIX}_{mode}_{tag}_valid_samples.csv"
FAILURE_CSV="$RUN_ROOT/{PREFIX}_{mode}_{tag}_failure_manifest.csv"
EVAL_A="$RUN_ROOT/promptA_condition_eval.csv"
EVAL_B="$RUN_ROOT/promptB_baseline_eval.csv"
GRAPH_A="$RUN_ROOT/graphs_a_condition"
GRAPH_B="$RUN_ROOT/graphs_b_baseline"
GRAPH_B_UNIQUE="$RUN_ROOT/graphs_b_baseline_unique"
META_A="$RUN_ROOT/answer_aligned_meta_a.csv"
META_B="$RUN_ROOT/answer_aligned_meta_b.csv"
B_UNIQUE_EVAL="$RUN_ROOT/promptB_baseline_unique_eval.csv"
B_UNIQUE_SELECTED="$RUN_ROOT/baseline_unique_selected.csv"
B_COPY_MAP="$RUN_ROOT/baseline_copy_map.csv"
DECISION_JSON="$STAGE/{PREFIX}_{mode}_{tag}_decision.json"

write_decision() {{
  local status="$1"
  local failure_type="$2"
  local failed_step="$3"
  .venv/bin/python - "$status" "$failure_type" "$failed_step" <<'PY'
import csv, json, sys
from pathlib import Path
status, failure_type, failed_step = sys.argv[1:4]
stage = Path("{REMOTE_STAGE}")
run_root = Path("{run_root}")
compare_dir = Path("{compare_dir}")
prefix = "{PREFIX}_{mode}_{tag}"
def count_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))
graph_a = sorted(str(path) for path in (run_root / "graphs_a_condition").glob("*.pt"))
graph_b = sorted(str(path) for path in (run_root / "graphs_b_baseline").glob("*.pt"))
valid_count = count_csv(run_root / f"{{prefix}}_valid_samples.csv")
payload = {{
    "status": status,
    "failure_type": failure_type,
    "failed_step": failed_step,
    "mode": "{mode}",
    "tag": "{tag}",
    "bucket": "{bucket}",
    "interpretation": "Gemma Stage6 prompt/text/CoT source-tracing counterpart. A=condition prompt, B=B_direct/original baseline.",
    "counts": {{
        "eval_a_rows": count_csv(run_root / "promptA_condition_eval.csv"),
        "eval_b_rows": count_csv(run_root / "promptB_baseline_eval.csv"),
        "valid_sample_rows": valid_count,
        "failure_rows": count_csv(run_root / f"{{prefix}}_failure_manifest.csv"),
        "meta_a_rows": count_csv(run_root / "answer_aligned_meta_a.csv"),
        "meta_b_rows": count_csv(run_root / "answer_aligned_meta_b.csv"),
        "sample_compare_rows": count_csv(compare_dir / "sample_compare_controlled.csv"),
        "nodes_detailed_rows": count_csv(compare_dir / "nodes_detailed_controlled.csv"),
        "edges_detailed_rows": count_csv(compare_dir / "edges_detailed_controlled.csv"),
        "graph_a_files": len(graph_a),
        "graph_b_files": len(graph_b),
        "graph_success_rate_vs_valid": (min(len(graph_a), len(graph_b)) / valid_count) if valid_count else 0.0,
    }},
    "artifacts": {{
        "eval_a": str(run_root / "promptA_condition_eval.csv"),
        "eval_b": str(run_root / "promptB_baseline_eval.csv"),
        "valid_samples": str(run_root / f"{{prefix}}_valid_samples.csv"),
        "failure_manifest": str(run_root / f"{{prefix}}_failure_manifest.csv"),
        "meta_a": str(run_root / "answer_aligned_meta_a.csv"),
        "meta_b": str(run_root / "answer_aligned_meta_b.csv"),
        "compare_sample": str(compare_dir / "sample_compare_controlled.csv"),
        "compare_nodes": str(compare_dir / "nodes_detailed_controlled.csv"),
        "compare_edges": str(compare_dir / "edges_detailed_controlled.csv"),
    }},
}}
(stage / f"{{prefix}}_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}}

disk_free_gb() {{
  df -BG --output=avail /root/autodl-tmp | tail -n 1 | tr -dc '0-9'
}}

log_disk() {{
  local label="$1"
  echo "--- disk:$label ---"
  df -h /root/autodl-tmp
  du -sh "$RUN_ROOT" "$GRAPH_A" "$GRAPH_B_UNIQUE" "$GRAPH_B" "$COMPARE_DIR" 2>/dev/null || true
}}

require_free_gb() {{
  local label="$1"
  local required="$2"
  local free_gb
  free_gb=$(disk_free_gb)
  echo "[disk_gate] label=$label free_gb=$free_gb required_gb=$required"
  if [ -z "$free_gb" ] || [ "$free_gb" -lt "$required" ]; then
    write_decision "blocked" "disk_free_below_threshold" "$label"
    exit 75
  fi
}}

run_step() {{
  local name="$1"
  local failure_type="$2"
  shift 2
  echo "--- $name ---"
  log_disk "$name:before"
  "$@"
  local code=$?
  log_disk "$name:after"
  if [ "$code" -ne 0 ]; then
    write_decision "blocked" "$failure_type" "$name"
    exit "$code"
  fi
}}

run_attr_step() {{
  local name="$1"
  local graph_dir="$2"
  shift 2
  echo "--- $name ---"
  log_disk "$name:before"
  "$@"
  local code=$?
  log_disk "$name:after"
  local pt_count=0
  if [ -d "$graph_dir" ]; then
    pt_count=$(find "$graph_dir" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')
  fi
  echo "[attr] $name exit=$code graph_pt_count=$pt_count"
  if [ "$pt_count" -lt 2 ]; then
    write_decision "blocked" "answer_aligned_attribute_oom_or_graph_missing" "$name"
    if [ "$code" -eq 0 ]; then exit 75; else exit "$code"; fi
  fi
  if [ "$code" -ne 0 ]; then
    echo "[warn] $name had per-sample attribution failures; continuing because graph_pt_count=$pt_count"
  fi
}}

echo '--- Stage6 Gemma prompt/text/CoT source-tracing disk/gpu ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo "mode={mode} tag={tag} resume={str(resume).lower()} max_new_tokens={max_new_tokens} max_feature_nodes={max_feature_nodes}"
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

{cleanup}
mkdir -p "$RUN_ROOT" "$COMPARE_DIR"
log_disk "after_cleanup_before_preflight"
require_free_gb "preflight" "$MIN_FREE_GB"

run_step "py_compile_gemma_stage6_research_scripts" "local_or_remote_script_compile" \\
  .venv/bin/python -m py_compile \\
  scripts/research/run_batch_eval.py \\
  scripts/research/run_batch_answer_aligned_attribute.py \\
  scripts/research/trace_compare_ab_controlled.py

run_step "eval_A_condition" "eval_prompt_or_gemma_model_load" \\
  .venv/bin/python -u scripts/research/run_batch_eval.py \\
  --manifest "$MANIFEST_A" \\
  --output-csv "$EVAL_A" \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --max-new-tokens {max_new_tokens} \\
  --correct-rule strict_gold \\
  {eval_resume_flag} \\
  --log-every 5

run_step "eval_B_baseline" "eval_prompt_or_gemma_model_load" \\
  .venv/bin/python -u scripts/research/run_batch_eval.py \\
  --manifest "$MANIFEST_B" \\
  --output-csv "$EVAL_B" \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --max-new-tokens {max_new_tokens} \\
  --correct-rule strict_gold \\
  {eval_resume_flag} \\
  --log-every 5

run_step "build_valid_sample_csv" "eval_empty_generation_or_target_alignment" \\
  .venv/bin/python - "$EVAL_A" "$EVAL_B" "$BUCKET_CSV" "$VALID_BUCKET_CSV" "$FAILURE_CSV" <<'PY'
import csv, sys
from pathlib import Path
eval_a, eval_b, bucket_csv, out_csv, failure_csv = [Path(arg) for arg in sys.argv[1:6]]
def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
def usable(row):
    return bool((row.get("generated_text") or "").strip()) and not (row.get("error_message") or "").strip()
def reason(row, side):
    if not row:
        return f"missing_eval_{{side}}"
    if (row.get("error_message") or "").strip():
        return f"error_{{side}}:" + (row.get("error_message") or "").strip()
    if not (row.get("generated_text") or "").strip():
        return f"empty_generated_{{side}}"
    return ""
a = {{row["sample_id"]: row for row in read_rows(eval_a)}}
b = {{row["sample_id"]: row for row in read_rows(eval_b)}}
bucket_rows = read_rows(bucket_csv)
valid, failures = [], []
for row in bucket_rows:
    sid = row["sample_id"]
    a_row = a.get(sid, {{}})
    b_row = b.get(sid, {{}})
    a_ok = usable(a_row)
    b_ok = usable(b_row)
    if a_ok and b_ok:
        valid.append(row)
    else:
        failures.append({{
            "sample_id": sid,
            "bucket": row.get("bucket", ""),
            "stage6_original_sample_id": row.get("stage6_original_sample_id", ""),
            "stage6_prompt_family": row.get("stage6_prompt_family", ""),
            "stage6_question_variant": row.get("stage6_question_variant", ""),
            "answer_text": row.get("answer_text", ""),
            "usable_a": "1" if a_ok else "0",
            "usable_b": "1" if b_ok else "0",
            "failure_a": reason(a_row, "a"),
            "failure_b": reason(b_row, "b"),
            "generated_a": a_row.get("generated_text", ""),
            "generated_b": b_row.get("generated_text", ""),
        }})
out_csv.parent.mkdir(parents=True, exist_ok=True)
valid_fields = list(bucket_rows[0].keys()) if bucket_rows else ["sample_id", "bucket"]
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=valid_fields)
    writer.writeheader()
    for row in valid:
        writer.writerow({{key: row.get(key, "") for key in valid_fields}})
failure_fields = [
    "sample_id", "bucket", "stage6_original_sample_id", "stage6_prompt_family",
    "stage6_question_variant", "answer_text", "usable_a", "usable_b",
    "failure_a", "failure_b", "generated_a", "generated_b",
]
with failure_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=failure_fields)
    writer.writeheader()
    writer.writerows(failures)
print(f"[valid] nonempty in both prompts={{len(valid)}}/{{len(bucket_rows)}}")
print(f"[failures] rows={{len(failures)}} -> {{failure_csv}}")
if len(valid) < 2:
    raise SystemExit("fewer than 2 valid samples after eval filtering")
PY

run_attr_step "attribute_A_condition_gold_answer" "$GRAPH_A" \\
  .venv/bin/python -u scripts/research/run_batch_answer_aligned_attribute.py \\
  --eval-csv "$EVAL_A" \\
  --output-dir "$GRAPH_A" \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --selected-csv "$VALID_BUCKET_CSV" \\
  --answer-source gold \\
  --metadata-csv "$META_A" \\
  --max-feature-nodes {max_feature_nodes} \\
  --retry-feature-nodes {retry_feature_nodes} \\
  --exec-mode subprocess

run_step "build_unique_B_baseline_graph_queue" "baseline_dedup_queue_build" \\
  .venv/bin/python - "$EVAL_B" "$VALID_BUCKET_CSV" "$B_UNIQUE_EVAL" "$B_UNIQUE_SELECTED" "$B_COPY_MAP" <<'PY'
import csv, sys
from pathlib import Path
eval_b, valid_csv, unique_eval, unique_selected, copy_map = [Path(arg) for arg in sys.argv[1:6]]
def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
eval_rows = {{row["sample_id"]: row for row in read_rows(eval_b)}}
valid_rows = read_rows(valid_csv)
first_by_original = {{}}
copy_rows = []
for row in valid_rows:
    sid = row["sample_id"]
    original = row.get("stage6_original_sample_id") or sid.split("__", 1)[0]
    first_by_original.setdefault(original, sid)
for row in valid_rows:
    sid = row["sample_id"]
    original = row.get("stage6_original_sample_id") or sid.split("__", 1)[0]
    copy_rows.append({{"source_sample_id": first_by_original[original], "target_sample_id": sid, "stage6_original_sample_id": original}})
selected_ids = set(first_by_original.values())
unique_eval_rows = [eval_rows[sid] for sid in selected_ids if sid in eval_rows]
unique_selected_rows = [row for row in valid_rows if row["sample_id"] in selected_ids]
for path, rows in [(unique_eval, unique_eval_rows), (unique_selected, unique_selected_rows), (copy_map, copy_rows)]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        continue
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[baseline_dedup] unique_originals={{len(selected_ids)}} valid_conditions={{len(valid_rows)}} unique_eval_rows={{len(unique_eval_rows)}}")
if len(unique_eval_rows) < 2:
    raise SystemExit("fewer than 2 unique baseline rows")
PY

run_attr_step "attribute_B_baseline_gold_answer_unique" "$GRAPH_B_UNIQUE" \\
  .venv/bin/python -u scripts/research/run_batch_answer_aligned_attribute.py \\
  --eval-csv "$B_UNIQUE_EVAL" \\
  --output-dir "$GRAPH_B_UNIQUE" \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --selected-csv "$B_UNIQUE_SELECTED" \\
  --answer-source gold \\
  --metadata-csv "$META_B" \\
  --max-feature-nodes {max_feature_nodes} \\
  --retry-feature-nodes {retry_feature_nodes} \\
  --exec-mode subprocess

run_step "copy_unique_B_baseline_graphs_to_conditions" "baseline_graph_copy_missing" \\
  .venv/bin/python - "$GRAPH_B_UNIQUE" "$GRAPH_B" "$B_COPY_MAP" <<'PY'
import csv, os, sys
from pathlib import Path
src_dir, dst_dir, copy_map = [Path(arg) for arg in sys.argv[1:4]]
dst_dir.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(copy_map.open("r", encoding="utf-8", newline="")))
copied = 0
linked = 0
missing = []
link_errors = []
for row in rows:
    src = src_dir / f"{{row['source_sample_id']}}.pt"
    dst = dst_dir / f"{{row['target_sample_id']}}.pt"
    if not src.exists():
        missing.append(str(src))
        continue
    if src.resolve() != dst.resolve():
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            try:
                dst.symlink_to(src)
                linked += 1
            except OSError as exc:
                link_errors.append(f"{{src}} -> {{dst}}: {{exc}}")
    copied += 1
print(f"[baseline_dedup] materialized={{copied}} linked_or_symlinked={{linked}} missing={{len(missing)}} link_errors={{len(link_errors)}} dst={{dst_dir}}")
if missing:
    print("\\n".join(missing[:20]))
    raise SystemExit("missing unique baseline graph(s)")
if link_errors:
    print("\\n".join(link_errors[:20]))
    raise SystemExit("hardlink/symlink failed; refusing to copy large graph files")
PY

run_step "trace_compare_ab_controlled" "compare_format_or_graph_missing" \\
  .venv/bin/python -u scripts/research/trace_compare_ab_controlled.py \\
  --pt-dir-a "$GRAPH_A" \\
  --pt-dir-b "$GRAPH_B" \\
  --bucket-csv "$VALID_BUCKET_CSV" \\
  --out-dir "$COMPARE_DIR" \\
  --buckets "$BUCKET" \\
  --per-bucket 999 \\
  --topk-per-node 16 \\
  --beam-per-depth 16 \\
  --coverage 0.85 \\
  --max-depth 4 \\
  --min-abs-weight 0

write_decision "pass" "" ""
"""


def _fetch_one(sftp, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        sftp.get(remote, str(local))
        print(f"fetched {local.name}", flush=True)
        return True
    except FileNotFoundError:
        print(f"missing {remote}", flush=True)
        return False


def _fetch_outputs(sftp, mode: str, tag: str) -> None:
    run_root = f"{REMOTE_STAGE}/source_tracing_{mode}_{tag}"
    compare_dir = f"{run_root}/compare"
    prefix = f"{PREFIX}_{mode}_{tag}"
    mapping = {
        f"{REMOTE_STAGE}/{prefix}_decision.json": LOCAL_CROSS / f"{prefix}_decision.json",
        f"{run_root}/promptA_condition_eval.csv": LOCAL_CROSS / f"{prefix}_eval_A_condition.csv",
        f"{run_root}/promptB_baseline_eval.csv": LOCAL_CROSS / f"{prefix}_eval_B_baseline.csv",
        f"{run_root}/{prefix}_valid_samples.csv": LOCAL_CROSS / f"{prefix}_valid_samples.csv",
        f"{run_root}/{prefix}_failure_manifest.csv": LOCAL_CROSS / f"{prefix}_failure_manifest.csv",
        f"{run_root}/answer_aligned_meta_a.csv": LOCAL_CROSS / f"{prefix}_meta_a.csv",
        f"{run_root}/answer_aligned_meta_b.csv": LOCAL_CROSS / f"{prefix}_meta_b.csv",
        f"{compare_dir}/sample_compare_controlled.csv": LOCAL_CROSS / f"{prefix}_sample_compare_controlled.csv",
        f"{compare_dir}/bucket_summary_controlled.csv": LOCAL_CROSS / f"{prefix}_bucket_summary_controlled.csv",
        f"{compare_dir}/nodes_detailed_controlled.csv": LOCAL_CROSS / f"{prefix}_nodes_detailed_controlled.csv",
        f"{compare_dir}/edges_detailed_controlled.csv": LOCAL_CROSS / f"{prefix}_edges_detailed_controlled.csv",
    }
    fetched = 0
    for remote, local in mapping.items():
        fetched += int(_fetch_one(sftp, remote, local))
    print(f"fetch complete: fetched={fetched} requested={len(mapping)}", flush=True)


def _status_command(mode: str, tag: str) -> str:
    return f"""
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'stage6_gemma_prompt_text_cot|run_batch_answer_aligned_attribute.py|trace_compare_ab_controlled.py' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{PREFIX}_{mode}_{tag}_* 2>/dev/null || true
ls -lh {REMOTE_STAGE}/source_tracing_{mode}_{tag} 2>/dev/null || true
echo DISK
df -h /root/autodl-tmp
du -sh {REMOTE_STAGE}/source_tracing_{mode}_{tag} {REMOTE_STAGE}/source_tracing_{mode}_{tag}/graphs_a_condition {REMOTE_STAGE}/source_tracing_{mode}_{tag}/graphs_b_baseline_unique {REMOTE_STAGE}/source_tracing_{mode}_{tag}/graphs_b_baseline {REMOTE_STAGE}/source_tracing_{mode}_{tag}/compare 2>/dev/null || true
du -sh {REMOTE_STAGE}/source_tracing_{mode}_{tag}/shards {REMOTE_STAGE}/source_tracing_{mode}_{tag}/shards/shard_* 2>/dev/null | tail -40 || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*gemma_prompt_text_cot* 2>/dev/null || true
echo LOG_TAIL
latest_log=$(ls -t {REMOTE_STAGE}/logs/gemma_prompt_text_cot_{mode}_*.log 2>/dev/null | head -n 1 || true)
if [ -n "$latest_log" ]; then
  echo "$latest_log"
  tail -n 100 "$latest_log"
fi
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage6 Gemma prompt/text/CoT source-tracing counterpart.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="gemmaprompt_v1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-feature-nodes", type=int, default=8)
    parser.add_argument("--allow-large-graphs", action="store_true")
    parser.add_argument("--sharded-streaming", action="store_true")
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--cleanup-graphs-after-compare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--cleanup-failed-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    min_free_gb = 0
    if not (args.status or args.fetch_only or args.cleanup_failed_run):
        try:
            min_free_gb = _min_free_gb(
                args.mode,
                args.max_feature_nodes,
                args.allow_large_graphs,
                args.sharded_streaming,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.mode == "full" and not args.sharded_streaming and not args.allow_large_graphs:
            # The remote will enforce the 80G gate as well. This local guard keeps the
            # daytime path explicit: normal full is allowed only as the compact gated run.
            pass
        if args.shard_count < 1:
            parser.error("--shard-count must be >= 1")
    if args.dry_run:
        print(
            {
                "mode": args.mode,
                "tag": args.tag,
                "max_feature_nodes": args.max_feature_nodes,
                "allow_large_graphs": args.allow_large_graphs,
                "sharded_streaming": args.sharded_streaming,
                "shard_count": args.shard_count,
                "cleanup_graphs_after_compare": args.cleanup_graphs_after_compare,
                "min_free_gb": min_free_gb,
                "run_root": f"{REMOTE_STAGE}/source_tracing_{args.mode}_{args.tag}",
            }
        )
        return 0

    a_path, b_path, manifest_path = _manifest_paths(args.tag)
    if not a_path.exists() or not b_path.exists() or not manifest_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "build_stage6_gemma_prompt_text_pack.py"),
                "--tag",
                args.tag,
            ],
            check=True,
        )

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)

    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.mode, args.tag))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    if args.cleanup_failed_run:
        _assert_local_cleanup_evidence(args.mode, args.tag)
        _stdin, stdout, stderr = client.exec_command(_cleanup_failed_run_command(args.mode, args.tag))
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        print(out)
        if err:
            print(err, file=sys.stderr)
        client.close()
        if code != 0:
            raise SystemExit(code)
        return 0

    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch_outputs(sftp, args.mode, args.tag)
        sftp.close()
        client.close()
        if not args.skip_analyze:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "local" / "analyze_stage6_gemma_prompt_text_cot.py"),
                    "--mode",
                    args.mode,
                    "--tag",
                    args.tag,
                ],
                check=False,
            )
        return 0

    for script_name in [
        "run_batch_eval.py",
        "run_batch_answer_aligned_attribute.py",
        "trace_compare_ab_controlled.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    _upload_pack(base, sftp, args.tag, args.mode)
    remote_script = f"{REMOTE_STAGE}/run_stage6_gemma_prompt_text_cot_{args.mode}_{args.tag}.sh"
    script_text = (
        _remote_sharded_script(
            args.mode,
            args.tag,
            args.resume,
            args.max_new_tokens,
            args.max_feature_nodes,
            min_free_gb,
            args.shard_count,
            args.cleanup_graphs_after_compare,
        )
        if args.sharded_streaming
        else _remote_script(
            args.mode,
            args.tag,
            args.resume,
            args.max_new_tokens,
            args.max_feature_nodes,
            min_free_gb,
        )
    )
    with sftp.file(remote_script, "w") as handle:
        handle.write(script_text.replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        remote_log_dir = f"{REMOTE_STAGE}/logs"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log = f"{remote_log_dir}/gemma_prompt_text_cot_{args.mode}_{stamp}.log"
        cmd = f"mkdir -p {remote_log_dir}; nohup bash {remote_script} > {remote_log} 2>&1 < /dev/null & echo $!"
        _stdin, stdout, stderr = client.exec_command(cmd)
        pid = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        print({"detached_remote_pid": pid, "remote_log": remote_log, "remote_script": remote_script})
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
    stdout.channel.settimeout(0.0)
    stderr.channel.settimeout(0.0)
    start = time.time()
    timed_out = False
    while not stdout.channel.exit_status_ready():
        try:
            data = stdout.channel.recv(8192)
            if data:
                print(data.decode("utf-8", errors="replace"), end="")
        except socket.timeout:
            pass
        try:
            data = stderr.channel.recv_stderr(8192)
            if data:
                print(data.decode("utf-8", errors="replace"), end="")
        except socket.timeout:
            pass
        if time.time() - start > args.timeout_seconds:
            timed_out = True
            stdout.channel.close()
            break
        time.sleep(0.5)
    exit_status = 124 if timed_out else stdout.channel.recv_exit_status()
    for stream, is_err in [(stdout, False), (stderr, True)]:
        try:
            data = stream.read()
            if data:
                print(data.decode("utf-8", errors="replace"), end="", file=sys.stderr if is_err else sys.stdout)
        except Exception:
            pass
    sftp = client.open_sftp()
    _fetch_outputs(sftp, args.mode, args.tag)
    sftp.close()
    client.close()
    if not args.skip_analyze:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage6_gemma_prompt_text_cot.py"),
                "--mode",
                args.mode,
                "--tag",
                args.tag,
            ],
            check=False,
        )
    if exit_status != 0:
        raise SystemExit(exit_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
