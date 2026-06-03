#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import posixpath
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage3_gemma_paperpack"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _suffix(pack: str, mode: str) -> str:
    return f"{pack}_{mode}"


def _bucket_name(pack: str, mode: str) -> str:
    return "paperpack_smoke" if mode == "smoke" else f"paperpack_{pack}_{mode}"


def _run_root(pack: str, mode: str) -> str:
    return f"{REMOTE_STAGE}/source_tracing_{pack}_{mode}"


def _compare_dir(pack: str, mode: str) -> str:
    return f"{_run_root(pack, mode)}/compare"


def _manifest_names(pack: str, mode: str) -> tuple[str, str]:
    if mode == "smoke":
        return "stage3_gemma_eval_smoke_B_direct.csv", "stage3_gemma_eval_smoke_D_visual_only.csv"
    return f"stage3_gemma_eval_{pack}_B_direct.csv", f"stage3_gemma_eval_{pack}_D_visual_only.csv"


def _remote_image_path(image_filename: str) -> str:
    return f"{REMOTE_ASSETS}/images/{Path(image_filename).name}"


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _remoteize_manifest_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Path]]:
    image_paths: dict[str, Path] = {}
    out: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        image_path = Path(item["image_path"])
        image_name = Path(item["image_filename"]).name
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image_paths[image_name] = image_path
        item["image_path"] = _remote_image_path(image_name)
        out.append(item)
    return out, image_paths


def _build_bucket_rows(rows: list[dict[str, str]], bucket: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        out.append(
            {
                "sample_id": sample_id,
                "bucket": bucket,
                "question_text": row.get("question_text", ""),
                "answer_text": row.get("answer_text", row.get("gold_answer", "")),
                "reasoning_operation": row.get("reasoning_operation", ""),
                "image_dependence_tier": row.get("image_dependence_tier", ""),
                "paperpack_source": row.get("paperpack_source", ""),
            }
        )
    return out


def _upload_manifests_and_images(base, sftp, pack: str, mode: str, limit: int) -> None:
    b_name, d_name = _manifest_names(pack, mode)
    uploaded_images: dict[str, Path] = {}
    manifest_rows: dict[str, list[dict[str, str]]] = {}
    for name in [b_name, d_name]:
        local = LOCAL_CROSS / name
        rows = _read_csv(local)
        if not rows:
            raise ValueError(f"empty manifest: {local}")
        if limit > 0:
            rows = rows[:limit]
        remote_rows, image_paths = _remoteize_manifest_rows(rows)
        manifest_rows[name] = remote_rows
        uploaded_images.update(image_paths)
        _put_text(base, sftp, f"{REMOTE_STAGE}/{name}", _csv_text(remote_rows, list(remote_rows[0].keys())))
        print(f"uploaded remoteized {name} rows={len(remote_rows)}", flush=True)

    bucket_rows = _build_bucket_rows(manifest_rows[d_name], _bucket_name(pack, mode))
    bucket_remote_name = f"stage3_gemma_source_tracing_{_suffix(pack, mode)}_manifest.csv"
    _put_text(
        base,
        sftp,
        f"{REMOTE_STAGE}/{bucket_remote_name}",
        _csv_text(
            bucket_rows,
            [
                "sample_id",
                "bucket",
                "question_text",
                "answer_text",
                "reasoning_operation",
                "image_dependence_tier",
                "paperpack_source",
            ],
        ),
    )
    print(f"uploaded {bucket_remote_name} rows={len(bucket_rows)}", flush=True)

    for image_name, image_path in sorted(uploaded_images.items()):
        base._put_file(sftp, image_path, f"{REMOTE_ASSETS}/images/{image_name}")
    print(f"uploaded images={len(uploaded_images)}", flush=True)


def _remote_script(pack: str, mode: str, limit: int, resume: bool) -> str:
    suffix = _suffix(pack, mode)
    bucket = _bucket_name(pack, mode)
    run_root = _run_root(pack, mode)
    compare_dir = _compare_dir(pack, mode)
    b_name, d_name = _manifest_names(pack, mode)
    eval_resume_flag = "--resume" if resume else "--no-resume"
    eval_limit_arg = f"--limit {limit}" if limit > 0 else ""
    max_samples_arg = f"--max-samples {limit}" if limit > 0 else ""
    per_bucket = max(999, limit if limit > 0 else 999)
    cleanup_block = (
        f"""case "$RUN_ROOT" in
  "$STAGE"/*) rm -rf "$RUN_ROOT" ;;
  *) echo "unsafe RUN_ROOT=$RUN_ROOT"; exit 99 ;;
esac
"""
        if not resume
        else 'echo "[resume] preserving RUN_ROOT=$RUN_ROOT"\n'
    )

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
BUCKET_CSV="$STAGE/stage3_gemma_source_tracing_{suffix}_manifest.csv"
VALID_BUCKET_CSV="$RUN_ROOT/stage3_gemma_source_tracing_{suffix}_valid_samples.csv"
FAILURE_CSV="$RUN_ROOT/stage3_gemma_source_tracing_{suffix}_failure_manifest.csv"
MANIFEST_B="$STAGE/{b_name}"
MANIFEST_D="$STAGE/{d_name}"
EVAL_A="$RUN_ROOT/promptA_D_visual_only_eval.csv"
EVAL_B="$RUN_ROOT/promptB_B_direct_eval.csv"
GRAPH_A="$RUN_ROOT/graphs_a_D_visual_only"
GRAPH_B="$RUN_ROOT/graphs_b_B_direct"
META_A="$RUN_ROOT/answer_aligned_meta_a.csv"
META_B="$RUN_ROOT/answer_aligned_meta_b.csv"
INTERVENTION_CSV="$RUN_ROOT/intervention_smoke_{suffix}.csv"
DECISION_JSON="$STAGE/stage3_gemma_source_tracing_{suffix}_decision.json"

write_decision() {{
  local status="$1"
  local failure_type="$2"
  local failed_step="$3"
  .venv/bin/python - "$status" "$failure_type" "$failed_step" <<'PY'
import csv
import json
import sys
from pathlib import Path

status, failure_type, failed_step = sys.argv[1:4]
stage = Path("{REMOTE_STAGE}")
run_root = Path("{run_root}")
compare_dir = Path("{compare_dir}")
suffix = "{suffix}"

def count_csv(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))

graph_a = sorted(str(path) for path in (run_root / "graphs_a_D_visual_only").glob("*.pt"))
graph_b = sorted(str(path) for path in (run_root / "graphs_b_B_direct").glob("*.pt"))
valid_count = count_csv(run_root / f"stage3_gemma_source_tracing_{{suffix}}_valid_samples.csv")
graph_success = min(len(graph_a), len(graph_b)) / valid_count if valid_count else 0.0
payload = {{
    "status": status,
    "failure_type": failure_type,
    "failed_step": failed_step,
    "pack": "{pack}",
    "mode": "{mode}",
    "bucket": "{bucket}",
    "interpretation": "Gemma paperpack source-tracing run; full mode is confirmatory input, smoke mode is engineering feasibility only.",
    "remote_run_root": str(run_root),
    "remote_compare_dir": str(compare_dir),
    "counts": {{
        "eval_a_rows": count_csv(run_root / "promptA_D_visual_only_eval.csv"),
        "eval_b_rows": count_csv(run_root / "promptB_B_direct_eval.csv"),
        "valid_sample_rows": valid_count,
        "failure_rows": count_csv(run_root / f"stage3_gemma_source_tracing_{{suffix}}_failure_manifest.csv"),
        "meta_a_rows": count_csv(run_root / "answer_aligned_meta_a.csv"),
        "meta_b_rows": count_csv(run_root / "answer_aligned_meta_b.csv"),
        "sample_compare_rows": count_csv(compare_dir / "sample_compare_controlled.csv"),
        "nodes_detailed_rows": count_csv(compare_dir / "nodes_detailed_controlled.csv"),
        "edges_detailed_rows": count_csv(compare_dir / "edges_detailed_controlled.csv"),
        "intervention_rows": count_csv(run_root / f"intervention_smoke_{{suffix}}.csv"),
        "graph_a_files": len(graph_a),
        "graph_b_files": len(graph_b),
        "graph_success_rate_vs_valid": graph_success,
    }},
    "graph_files": {{"a": graph_a, "b": graph_b}},
    "artifacts": {{
        "eval_a": str(run_root / "promptA_D_visual_only_eval.csv"),
        "eval_b": str(run_root / "promptB_B_direct_eval.csv"),
        "valid_samples": str(run_root / f"stage3_gemma_source_tracing_{{suffix}}_valid_samples.csv"),
        "failure_manifest": str(run_root / f"stage3_gemma_source_tracing_{{suffix}}_failure_manifest.csv"),
        "meta_a": str(run_root / "answer_aligned_meta_a.csv"),
        "meta_b": str(run_root / "answer_aligned_meta_b.csv"),
        "compare_sample": str(compare_dir / "sample_compare_controlled.csv"),
        "compare_nodes": str(compare_dir / "nodes_detailed_controlled.csv"),
        "intervention": str(run_root / f"intervention_smoke_{{suffix}}.csv"),
    }},
}}
(stage / f"stage3_gemma_source_tracing_{{suffix}}_decision.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}}

run_step() {{
  local name="$1"
  local failure="$2"
  shift 2
  echo "--- $name ---"
  "$@"
  local code=$?
  if [ "$code" -ne 0 ]; then
    echo "[failed] $name exit=$code"
    write_decision "blocked" "$failure" "$name"
    exit "$code"
  fi
}}

run_attr_step() {{
  local name="$1"
  local graph_dir="$2"
  shift 2
  echo "--- $name ---"
  "$@"
  local code=$?
  local pt_count=0
  if [ -d "$graph_dir" ]; then
    pt_count=$(find "$graph_dir" -maxdepth 1 -name '*.pt' | wc -l | tr -d ' ')
  fi
  echo "[attr] $name exit=$code graph_pt_count=$pt_count"
  if [ "$pt_count" -lt 2 ]; then
    echo "[failed] $name produced too few graph files"
    write_decision "blocked" "answer_aligned_attribute_oom_or_graph_missing" "$name"
    exit "$code"
  fi
  if [ "$code" -ne 0 ]; then
    echo "[warn] $name had per-sample attribution failures; continuing because graph_pt_count=$pt_count"
  fi
}}

echo '--- Stage3 Gemma paperpack source-tracing disk/gpu ---'
echo "pack={pack} mode={mode} suffix={suffix} limit={limit} resume={str(resume).lower()}"
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

{cleanup_block}
mkdir -p "$RUN_ROOT" "$COMPARE_DIR"

run_step "py_compile_gemma_research_scripts" "local_or_remote_script_compile" \\
  .venv/bin/python -m py_compile \\
  scripts/research/run_batch_eval.py \\
  scripts/research/run_batch_answer_aligned_attribute.py \\
  scripts/research/trace_compare_ab_controlled.py \\
  scripts/research/run_answer_aligned_intervention_smoke.py

run_step "eval_A_D_visual_only" "eval_prompt_or_gemma_model_load" \\
  .venv/bin/python -u scripts/research/run_batch_eval.py \\
  --manifest "$MANIFEST_D" \\
  --output-csv "$EVAL_A" \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --max-new-tokens 16 \\
  --correct-rule strict_gold \\
  {eval_limit_arg} \\
  {eval_resume_flag} \\
  --log-every 5

run_step "eval_B_B_direct" "eval_prompt_or_gemma_model_load" \\
  .venv/bin/python -u scripts/research/run_batch_eval.py \\
  --manifest "$MANIFEST_B" \\
  --output-csv "$EVAL_B" \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --max-new-tokens 16 \\
  --correct-rule strict_gold \\
  {eval_limit_arg} \\
  {eval_resume_flag} \\
  --log-every 5

run_step "build_valid_sample_csv" "eval_empty_generation_or_target_alignment" \\
  .venv/bin/python - "$EVAL_A" "$EVAL_B" "$BUCKET_CSV" "$VALID_BUCKET_CSV" "$FAILURE_CSV" <<'PY'
import csv
import sys
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
valid = []
failures = []
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
            "answer_text": row.get("answer_text", ""),
            "reasoning_operation": row.get("reasoning_operation", ""),
            "image_dependence_tier": row.get("image_dependence_tier", ""),
            "paperpack_source": row.get("paperpack_source", ""),
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
    "sample_id",
    "bucket",
    "answer_text",
    "reasoning_operation",
    "image_dependence_tier",
    "paperpack_source",
    "usable_a",
    "usable_b",
    "failure_a",
    "failure_b",
    "generated_a",
    "generated_b",
]
with failure_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=failure_fields)
    writer.writeheader()
    for row in failures:
        writer.writerow(row)

print(f"[valid] generated_text_nonempty_in_both_prompts={{len(valid)}}/{{len(bucket_rows)}} -> {{out_csv}}")
print(f"[failures] rows={{len(failures)}} -> {{failure_csv}}")
if len(valid) < 2:
    raise SystemExit("fewer than 2 valid samples after eval filtering")
PY

run_attr_step "attribute_A_gold_answer" "$GRAPH_A" \\
  .venv/bin/python -u scripts/research/run_batch_answer_aligned_attribute.py \\
  --eval-csv "$EVAL_A" \\
  --output-dir "$GRAPH_A" \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --selected-csv "$VALID_BUCKET_CSV" \\
  --answer-source gold \\
  --metadata-csv "$META_A" \\
  --max-feature-nodes 64 \\
  --retry-feature-nodes 48,32,16 \\
  --exec-mode subprocess

run_attr_step "attribute_B_gold_answer" "$GRAPH_B" \\
  .venv/bin/python -u scripts/research/run_batch_answer_aligned_attribute.py \\
  --eval-csv "$EVAL_B" \\
  --output-dir "$GRAPH_B" \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --selected-csv "$VALID_BUCKET_CSV" \\
  --answer-source gold \\
  --metadata-csv "$META_B" \\
  --max-feature-nodes 64 \\
  --retry-feature-nodes 48,32,16 \\
  --exec-mode subprocess

run_step "trace_compare_ab_controlled" "compare_format_or_graph_missing" \\
  .venv/bin/python -u scripts/research/trace_compare_ab_controlled.py \\
  --pt-dir-a "$GRAPH_A" \\
  --pt-dir-b "$GRAPH_B" \\
  --bucket-csv "$VALID_BUCKET_CSV" \\
  --out-dir "$COMPARE_DIR" \\
  --buckets "$BUCKET" \\
  --per-bucket {per_bucket} \\
  --topk-per-node 16 \\
  --beam-per-depth 16 \\
  --coverage 0.85 \\
  --max-depth 4 \\
  --min-abs-weight 0

run_step "answer_aligned_intervention_smoke" "intervention_position_or_model_load" \\
  .venv/bin/python -u scripts/research/run_answer_aligned_intervention_smoke.py \\
  --run-root "$RUN_ROOT" \\
  --compare-dir "$COMPARE_DIR" \\
  --bucket "$BUCKET" \\
  --run both \\
  --transcoder-set tianhux2/gemma3-4b-it-plt \\
  --sample-ids-csv "$VALID_BUCKET_CSV" \\
  {max_samples_arg} \\
  --top-features-per-sample 1 \\
  --max-pos-buffer 2 \\
  --out-csv "$INTERVENTION_CSV"

write_decision "pass" "" ""
"""


def _fetch_one(sftp, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        sftp.get(remote, str(local))
        print(f"fetched {local.name}", flush=True)
    except FileNotFoundError:
        print(f"missing {remote}", flush=True)


def _fetch_outputs(sftp, pack: str, mode: str) -> None:
    suffix = _suffix(pack, mode)
    run_root = _run_root(pack, mode)
    compare_dir = _compare_dir(pack, mode)
    mapping = {
        f"{REMOTE_STAGE}/stage3_gemma_source_tracing_{suffix}_decision.json": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_decision.json",
        f"{run_root}/promptA_D_visual_only_eval.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_eval_A_D_visual_only.csv",
        f"{run_root}/promptB_B_direct_eval.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_eval_B_B_direct.csv",
        f"{run_root}/stage3_gemma_source_tracing_{suffix}_valid_samples.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_valid_samples.csv",
        f"{run_root}/stage3_gemma_source_tracing_{suffix}_failure_manifest.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_failure_manifest.csv",
        f"{run_root}/answer_aligned_meta_a.csv": LOCAL_CROSS / f"stage3_gemma_source_tracing_{suffix}_meta_a.csv",
        f"{run_root}/answer_aligned_meta_b.csv": LOCAL_CROSS / f"stage3_gemma_source_tracing_{suffix}_meta_b.csv",
        f"{compare_dir}/sample_compare_controlled.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_sample_compare_controlled.csv",
        f"{compare_dir}/bucket_summary_controlled.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_bucket_summary_controlled.csv",
        f"{compare_dir}/nodes_detailed_controlled.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_nodes_detailed_controlled.csv",
        f"{compare_dir}/edges_detailed_controlled.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_edges_detailed_controlled.csv",
        f"{run_root}/intervention_smoke_{suffix}.csv": LOCAL_CROSS
        / f"stage3_gemma_source_tracing_{suffix}_intervention.csv",
    }
    for remote, local in mapping.items():
        _fetch_one(sftp, remote, local)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage3 Gemma3-PLT paperpack source tracing on AutoDL.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--limit", type=int, default=0, help="Optional row/sample cap for debugging; 0 means all rows.")
    parser.add_argument("--resume", action="store_true", help="Preserve remote run root and resume eval/graph work.")
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    args = parser.parse_args()

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username="root",
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    sftp = client.open_sftp()
    base._mkdir_p(sftp, REMOTE_STAGE)
    for script_name in [
        "run_batch_eval.py",
        "run_batch_answer_aligned_attribute.py",
        "trace_compare_ab_controlled.py",
        "run_answer_aligned_intervention_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    _upload_manifests_and_images(base, sftp, args.pack, args.mode, args.limit)

    suffix = _suffix(args.pack, args.mode)
    remote_script = f"{REMOTE_STAGE}/run_stage3_gemma_source_tracing_{suffix}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.pack, args.mode, args.limit, args.resume).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
    stdout.channel.settimeout(0.0)
    stderr.channel.settimeout(0.0)
    start = time.time()
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
        time.sleep(0.5)
        if time.time() - start > args.timeout_seconds:
            stdout.channel.close()
            raise TimeoutError("remote Stage3 Gemma source-tracing command exceeded timeout")

    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}", flush=True)
    sftp = client.open_sftp()
    _fetch_outputs(sftp, args.pack, args.mode)
    sftp.close()
    client.close()
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
