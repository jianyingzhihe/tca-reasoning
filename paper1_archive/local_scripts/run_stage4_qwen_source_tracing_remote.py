#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import socket
import sys
import time
import json
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_source_tracing"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _manifest_paths(pack: str) -> tuple[Path, Path]:
    if pack == "primary":
        return (
            LOCAL_PAPERPACK / "paperpack72_primary_manifest.csv",
            LOCAL_PAPERPACK / "paperpack72_primary_prompt_runs.csv",
        )
    if pack == "strict":
        return (
            LOCAL_PAPERPACK / "paperpack72_strict_sensitivity_manifest.csv",
            LOCAL_PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv",
        )
    raise ValueError(f"unsupported pack: {pack}")


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _put_if_missing(base, sftp, local: Path, remote: str) -> bool:
    try:
        sftp.stat(remote)
        return False
    except OSError:
        base._put_file(sftp, local, remote)
        return True


def _upload_pack_assets(base, sftp, pack: str, mode: str) -> None:
    sample_manifest, prompt_manifest = _manifest_paths(pack)
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    base._put_file(sftp, sample_manifest, f"{REMOTE_STAGE}/paperpack_{pack}_manifest.csv")
    base._put_file(sftp, prompt_manifest, f"{REMOTE_STAGE}/paperpack_{pack}_prompt_runs.csv")

    with sample_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    if mode == "smoke":
        with prompt_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            prompt_rows = list(csv.DictReader(handle))[:6]
        keep_ids = {row["sample_id"] for row in prompt_rows}
        sample_rows = [row for row in sample_rows if row["sample_id"] in keep_ids]

    uploaded = 0
    skipped = 0
    for row in sample_rows:
        image = Path(row["local_image_path"])
        image_name = Path(row["image_filename"]).name
        if _put_if_missing(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
            uploaded += 1
        else:
            skipped += 1
    print(
        f"uploaded Stage4 Qwen source-tracing {pack} assets: samples={len(sample_rows)} "
        f"files_uploaded={uploaded} files_skipped_existing={skipped}",
        flush=True,
    )


def _prefix(pack: str, mode: str, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
    return f"stage4_qwen_source_tracing_{pack}_{mode}{suffix}"


def _remote_script(
    pack: str,
    mode: str,
    tag: str = "",
    *,
    layer: int = 26,
    max_feature_nodes: int = 96,
    candidate_pool_size: int = 4096,
    compare_topk_per_node: int = 3,
    top_features_per_sample: int = 2,
    position_filter: str = "visual_answer",
    cleanup_run_root: bool = False,
) -> str:
    prefix = _prefix(pack, mode, tag)
    run_suffix = f"_{tag}" if tag else ""
    max_runs = "--max-runs 3" if mode == "smoke" else "--max-runs 0"
    intervention_limit = (
        f"--max-samples 3 --top-features-per-sample {top_features_per_sample}"
        if mode == "smoke"
        else f"--max-samples 0 --top-features-per-sample {top_features_per_sample}"
    )
    return f"""#!/usr/bin/env bash
set -e
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub
STAGE={REMOTE_STAGE}
ASSET_ROOT={REMOTE_ASSETS}
RUN_ROOT="$STAGE/source_tracing_{pack}_{mode}{run_suffix}"
PREFIX={prefix}
export STAGE PREFIX
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
SAMPLE_MANIFEST="$STAGE/paperpack_{pack}_manifest.csv"
RUN_MANIFEST="$STAGE/paperpack_{pack}_prompt_runs.csv"
mkdir -p "$RUN_ROOT" "$STAGE"

echo '--- Stage4 Qwen source tracing disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_qwen_answer_aligned_attribute.py \\
  scripts/research/run_qwen_answer_aligned_intervention_smoke.py \\
  scripts/research/trace_compare_ab_controlled.py \\
  scripts/research/analyze_stage4_qwen_source_tracing.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: Stage4 Qwen source tracing skipped'
  exit 3
fi

echo '--- Qwen D_visual_only answer-aligned graphs ---'
.venv/bin/python -u scripts/research/run_qwen_answer_aligned_attribute.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$RUN_MANIFEST" \\
  --prompt-name D_visual_only \\
  --image-root "$ASSET_ROOT/images" \\
  --out-dir "$RUN_ROOT/graphs_a_D_visual_only" \\
  --meta-csv "$STAGE/${{PREFIX}}_meta_a.csv" \\
  --summary-json "$STAGE/${{PREFIX}}_attribute_a.json" \\
  --layer {layer} \\
  --max-feature-nodes {max_feature_nodes} \\
  --candidate-pool-size {candidate_pool_size} \\
  --node-sign support \\
  --position-filter {position_filter} \\
  {max_runs}

echo '--- Qwen B_direct answer-aligned graphs ---'
.venv/bin/python -u scripts/research/run_qwen_answer_aligned_attribute.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$RUN_MANIFEST" \\
  --prompt-name B_direct \\
  --image-root "$ASSET_ROOT/images" \\
  --out-dir "$RUN_ROOT/graphs_b_B_direct" \\
  --meta-csv "$STAGE/${{PREFIX}}_meta_b.csv" \\
  --summary-json "$STAGE/${{PREFIX}}_attribute_b.json" \\
  --layer {layer} \\
  --max-feature-nodes {max_feature_nodes} \\
  --candidate-pool-size {candidate_pool_size} \\
  --node-sign support \\
  --position-filter {position_filter} \\
  {max_runs}

echo '--- Build valid A/B bucket CSV ---'
.venv/bin/python - <<'PY'
import csv, os
stage = os.environ["STAGE"]
prefix = os.environ["PREFIX"]
meta_a = os.path.join(stage, f"{prefix}_meta_a.csv")
meta_b = os.path.join(stage, f"{prefix}_meta_b.csv")
out = os.path.join(stage, f"{prefix}_valid_samples.csv")
def ok(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return {{r['sample_id']: r for r in csv.DictReader(f) if r.get('status') == 'ok'}}
a = ok(meta_a)
b = ok(meta_b)
ids = sorted(set(a) & set(b))
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['sample_id','bucket'])
    w.writeheader()
    for sid in ids:
        w.writerow({{'sample_id': sid, 'bucket': f'paperpack_{prefix}'}})
print(f"valid matched samples: {{len(ids)}}")
PY

echo '--- Controlled A/B compare ---'
.venv/bin/python -u scripts/research/trace_compare_ab_controlled.py \\
  --pt-dir-a "$RUN_ROOT/graphs_a_D_visual_only" \\
  --pt-dir-b "$RUN_ROOT/graphs_b_B_direct" \\
  --bucket-csv "$STAGE/${{PREFIX}}_valid_samples.csv" \\
  --out-dir "$RUN_ROOT/compare" \\
  --buckets "paperpack_${{PREFIX}}" \\
  --per-bucket 999 \\
  --target-logit-rank 0 \\
  --topk-per-node {compare_topk_per_node} \\
  --beam-per-depth 64 \\
  --coverage 0.95 \\
  --max-depth 8

cp "$RUN_ROOT/compare/sample_compare_controlled.csv" "$STAGE/${{PREFIX}}_sample_compare_controlled.csv"
cp "$RUN_ROOT/compare/nodes_detailed_controlled.csv" "$STAGE/${{PREFIX}}_nodes_detailed_controlled.csv"
cp "$RUN_ROOT/compare/edges_detailed_controlled.csv" "$STAGE/${{PREFIX}}_edges_detailed_controlled.csv"
cp "$RUN_ROOT/compare/bucket_summary_controlled.csv" "$STAGE/${{PREFIX}}_bucket_summary_controlled.csv" || true

echo '--- Qwen feature-node zeroing intervention smoke ---'
.venv/bin/python -u scripts/research/run_qwen_answer_aligned_intervention_smoke.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$RUN_MANIFEST" \\
  --nodes-csv "$STAGE/${{PREFIX}}_nodes_detailed_controlled.csv" \\
  --meta-a "$STAGE/${{PREFIX}}_meta_a.csv" \\
  --meta-b "$STAGE/${{PREFIX}}_meta_b.csv" \\
  --image-root "$ASSET_ROOT/images" \\
  --out-csv "$STAGE/${{PREFIX}}_intervention.csv" \\
  --summary-json "$STAGE/${{PREFIX}}_intervention.json" \\
  --layer {layer} \\
  --zeroing-modes subtract,add \\
  {intervention_limit}

echo '--- Stage4 Qwen source tracing analysis ---'
.venv/bin/python -u scripts/research/analyze_stage4_qwen_source_tracing.py \\
  --cross-dir "$STAGE" \\
  --pack {pack} \\
  --mode {mode} \\
  --out-prefix "$PREFIX"

if [ "{'1' if cleanup_run_root else '0'}" = "1" ]; then
  echo "--- Cleanup remote run root to preserve disk: $RUN_ROOT ---"
  rm -rf "$RUN_ROOT"
fi
"""


def _fetch_outputs(sftp, pack: str, mode: str, tag: str = "") -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(pack, mode, tag)
    names = [
        f"{prefix}_meta_a.csv",
        f"{prefix}_meta_b.csv",
        f"{prefix}_attribute_a.json",
        f"{prefix}_attribute_b.json",
        f"{prefix}_valid_samples.csv",
        f"{prefix}_sample_compare_controlled.csv",
        f"{prefix}_nodes_detailed_controlled.csv",
        f"{prefix}_edges_detailed_controlled.csv",
        f"{prefix}_bucket_summary_controlled.csv",
        f"{prefix}_intervention.csv",
        f"{prefix}_intervention.json",
        f"{prefix}_analysis_summary.csv",
        f"{prefix}_decision.json",
    ]
    for name in names:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def _status_command(pack: str, mode: str, tag: str) -> str:
    prefix = _prefix(pack, mode, tag)
    return f"""
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'run_qwen_answer_aligned|trace_compare_ab_controlled|run_qwen_answer_aligned_intervention|{prefix}' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{prefix}_* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*{prefix}* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4 Qwen source-tracing smoke/full job on AutoDL.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="", help="Optional artifact suffix, e.g. hookfix, to avoid overwriting prior runs.")
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--max-feature-nodes", type=int, default=96)
    parser.add_argument("--candidate-pool-size", type=int, default=4096)
    parser.add_argument("--compare-topk-per-node", type=int, default=3)
    parser.add_argument("--top-features-per-sample", type=int, default=2)
    parser.add_argument(
        "--position-filter",
        choices=["visual_answer", "visual_only", "answer_adjacent_only", "all"],
        default="visual_answer",
    )
    parser.add_argument("--cleanup-run-root", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true", help="Launch remote job under nohup and return immediately.")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch expected artifacts; do not upload or start a job.")
    parser.add_argument("--status", action="store_true", help="Print remote process/file status for this pack/mode/tag.")
    args = parser.parse_args()

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.pack, args.mode, args.tag))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err)
        client.close()
        return 0
    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch_outputs(sftp, args.pack, args.mode, args.tag)
        sftp.close()
        client.close()
        return 0
    for script_name in [
        "run_qwen_answer_aligned_attribute.py",
        "run_qwen_answer_aligned_intervention_smoke.py",
        "analyze_stage4_qwen_source_tracing.py",
        "trace_compare_ab_controlled.py",
        "run_cross_model_feature_intervention_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    _upload_pack_assets(base, sftp, args.pack, args.mode)

    script_suffix = f"_{args.tag}" if args.tag else ""
    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_source_tracing_{args.pack}_{args.mode}{script_suffix}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                args.pack,
                args.mode,
                args.tag,
                layer=args.layer,
                max_feature_nodes=args.max_feature_nodes,
                candidate_pool_size=args.candidate_pool_size,
                compare_topk_per_node=args.compare_topk_per_node,
                top_features_per_sample=args.top_features_per_sample,
                position_filter=args.position_filter,
                cleanup_run_root=args.cleanup_run_root,
            ).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log_dir = f"{REMOTE_STAGE}/logs"
        remote_log = f"{remote_log_dir}/{_prefix(args.pack, args.mode, args.tag)}_{stamp}.log"
        detach_cmd = f"mkdir -p {remote_log_dir}; nohup bash {remote_script} > {remote_log} 2>&1 < /dev/null & echo $!"
        _stdin, stdout, stderr = client.exec_command(detach_cmd)
        pid = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        print(json.dumps({"detached_remote_pid": pid, "remote_log": remote_log, "remote_script": remote_script}, indent=2), flush=True)
        if err:
            print(err, flush=True)
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
        time.sleep(0.5)
        if time.time() - start > args.timeout_seconds:
            timed_out = True
            stdout.channel.close()
            break

    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = 124 if timed_out else stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}", flush=True)
    sftp = client.open_sftp()
    _fetch_outputs(sftp, args.pack, args.mode, args.tag)
    sftp.close()
    client.close()
    if timed_out:
        raise TimeoutError("remote Stage4 Qwen source-tracing command exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
