#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_decisive_route"
REMOTE_SOURCE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_source_tracing"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
PRIMARY_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
STRICT_RUNS = PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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


def _upload_prompt_runs(base, sftp, runs_path: Path, remote_name: str) -> None:
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/exported_masks")
    base._put_file(sftp, runs_path, f"{REMOTE_STAGE}/{remote_name}")
    rows = list(csv.DictReader(runs_path.open("r", encoding="utf-8-sig", newline="")))
    uploaded = 0
    skipped = 0
    seen_images: set[str] = set()
    seen_masks: set[tuple[str, str]] = set()
    for row in rows:
        image = Path(row.get("local_image_path", ""))
        image_name = Path(row.get("image_filename", "")).name
        if image.exists() and image_name and image_name not in seen_images:
            seen_images.add(image_name)
            if _put_if_missing(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
                uploaded += 1
            else:
                skipped += 1
        stem = Path(image_name).stem
        for name in ["answer.png", "union.png", "shifted.png", "shuffled.png"]:
            key = (stem, name)
            if key in seen_masks:
                continue
            seen_masks.add(key)
            local_mask = Path(row.get("mask_dir", "")) / name
            if not local_mask.exists():
                continue
            if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{name}"):
                uploaded += 1
            else:
                skipped += 1
    print(f"uploaded {remote_name}: rows={len(rows)} uploaded={uploaded} skipped_existing={skipped}", flush=True)


def _prefix(pack: str, mode: str, layer: int, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"stage4_qwen_adapter_v4_{pack}_{mode}_L{layer}{suffix}"


def _source_prefix(pack: str, mode: str, source_tag: str) -> str:
    suffix = f"_{source_tag}" if source_tag else ""
    return f"stage4_qwen_source_tracing_{pack}_{mode}{suffix}"


def _remote_script(
    *,
    pack: str,
    mode: str,
    layer: int,
    tag: str,
    source_tag: str,
    top_per_prompt_run: int,
    main_per_prompt_run: int,
    allow_feature_fallback: bool,
    max_candidates: int,
) -> str:
    runs_name = f"paperpack72_{pack}_prompt_runs.csv"
    prefix = _prefix(pack, mode, layer, tag)
    source_prefix = _source_prefix(pack, mode, source_tag)
    source_intervention = f"{REMOTE_SOURCE_STAGE}/{source_prefix}_intervention.csv"
    evidence_candidates = f"{REMOTE_STAGE}/stage4_qwen_decisive_route_plt_{pack}_{mode}_L{layer}_candidates.csv"
    fallback_flag = "  --allow-feature-fallback \\\n" if allow_feature_fallback else ""
    max_candidates_flag = max_candidates if mode == "smoke" and max_candidates <= 0 else max_candidates
    if mode == "smoke" and max_candidates_flag <= 0:
        max_candidates_flag = 6
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
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
mkdir -p "$STAGE"

echo '--- Stage4 Qwen Adapter V4 disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_adapter_v4_route_probe.py \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: Adapter V4 skipped'
  exit 3
fi
if [ ! -f "{source_intervention}" ]; then
  echo 'SOURCE_TRACING_INTERVENTION_MISSING: {source_intervention}'
  exit 4
fi
if [ ! -f "{evidence_candidates}" ]; then
  echo 'PLT_EVIDENCE_CANDIDATES_MISSING: {evidence_candidates}'
  exit 5
fi

echo '--- Build Adapter V4 manifest ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_adapter_v4_route_probe.py \\
  --source-tracing-intervention "{source_intervention}" \\
  --evidence-discovery "{evidence_candidates}" \\
  --out-manifest "$STAGE/{prefix}_manifest.csv" \\
  --summary-json "$STAGE/{prefix}_manifest.json" \\
  --layer {layer} \\
  --hidden-layer 14 \\
  --top-per-prompt-run {top_per_prompt_run} \\
  --main-per-prompt-run {main_per_prompt_run} \\
{fallback_flag}  --max-clean-rank 10

echo '--- Adapter V4 zeroing controls ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{prefix}_manifest.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --work-dir "$STAGE/work_{prefix}_zeroing" \\
  --out-csv "$STAGE/{prefix}_zeroing_raw.csv" \\
  --summary-json "$STAGE/{prefix}_zeroing_run.json" \\
  --layer {layer} \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-candidates {max_candidates_flag}

echo '--- Adapter V4 grouped restore ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{prefix}_manifest.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{prefix}_group_raw.csv" \\
  --summary-json "$STAGE/{prefix}_group_run.json" \\
  --layer {layer} \\
  --top-ks 1,4,8,16,32,64 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-prompt-runs {"6" if mode == "smoke" else "0"}
"""


def _fetch(sftp, pack: str, mode: str, layer: int, tag: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(pack, mode, layer, tag)
    names = [
        f"{prefix}_manifest.csv",
        f"{prefix}_manifest.json",
        f"{prefix}_zeroing_raw.csv",
        f"{prefix}_zeroing_run.json",
        f"{prefix}_group_raw.csv",
        f"{prefix}_group_run.json",
    ]
    for name in names:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def _status_command(pack: str, mode: str, layer: int, tag: str) -> str:
    prefix = _prefix(pack, mode, layer, tag)
    return f"""
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'run_stage4_qwen_adapter_v4|run_stage4_qwen_causal_cutter|run_stage4_qwen_evidence_first|{prefix}' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{prefix}_* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*{prefix}* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-034 Qwen Adapter V4 route probe on AutoDL.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--tag", default="layer14")
    parser.add_argument("--source-tag", default="adapter_v4_L14")
    parser.add_argument("--top-per-prompt-run", type=int, default=4)
    parser.add_argument("--main-per-prompt-run", type=int, default=2)
    parser.add_argument("--allow-feature-fallback", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.pack, args.mode, args.layer, args.tag))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err)
        client.close()
        return 0

    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch(sftp, args.pack, args.mode, args.layer, args.tag)
        sftp.close()
        client.close()
        return 0

    for script in [
        "run_stage4_qwen_adapter_v4_route_probe.py",
        "run_stage4_qwen_causal_cutter_validation.py",
        "run_stage4_qwen_evidence_first_intervention.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
    ]:
        _upload_research_script(base, sftp, script)
    if args.pack == "primary":
        _upload_prompt_runs(base, sftp, PRIMARY_RUNS, "paperpack72_primary_prompt_runs.csv")
    else:
        _upload_prompt_runs(base, sftp, STRICT_RUNS, "paperpack72_strict_prompt_runs.csv")

    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_adapter_v4_{args.pack}_{args.mode}_L{args.layer}_{args.tag}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                pack=args.pack,
                mode=args.mode,
                layer=args.layer,
                tag=args.tag,
                source_tag=args.source_tag,
                top_per_prompt_run=args.top_per_prompt_run,
                main_per_prompt_run=args.main_per_prompt_run,
                allow_feature_fallback=args.allow_feature_fallback,
                max_candidates=args.max_candidates,
            ).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log_dir = f"{REMOTE_STAGE}/logs"
        prefix = _prefix(args.pack, args.mode, args.layer, args.tag)
        remote_log = f"{remote_log_dir}/{prefix}_{stamp}.log"
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
    _fetch(sftp, args.pack, args.mode, args.layer, args.tag)
    sftp.close()
    client.close()
    if not args.skip_analyze and exit_status == 0:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_adapter_v4.py"),
                "--mode",
                args.mode,
                "--layer",
                str(args.layer),
                "--tag",
                args.tag,
            ],
            check=False,
        )
    if timed_out:
        raise TimeoutError("remote Qwen Adapter V4 exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
