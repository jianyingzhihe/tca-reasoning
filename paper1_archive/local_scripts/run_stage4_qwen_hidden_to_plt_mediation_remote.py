#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_native_route"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
RUNS = {
    "primary": PAPERPACK / "paperpack72_primary_prompt_runs.csv",
    "strict": PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv",
}


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
            if local_mask.exists():
                if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{name}"):
                    uploaded += 1
                else:
                    skipped += 1
    print(f"uploaded {remote_name}: rows={len(rows)} uploaded={uploaded} skipped_existing={skipped}", flush=True)


def _prefix(pack: str, mode: str, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"stage4_qwen_native_route_hidden_to_plt_{pack}_{mode}{suffix}"


def _remote_script(pack: str, mode: str, tag: str, max_prompt_runs: int) -> str:
    prefix = _prefix(pack, mode, tag)
    runs_name = f"paperpack72_{pack}_prompt_runs.csv"
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

echo '--- Stage4-038 hidden-to-PLT disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_hidden_to_plt_mediation.py \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: hidden-to-PLT skipped'
  exit 3
fi

.venv/bin/python -u scripts/research/run_stage4_qwen_hidden_to_plt_mediation.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --prompt-runs "$STAGE/{runs_name}" \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{prefix}_raw.csv" \\
  --summary-json "$STAGE/{prefix}_run.json" \\
  --layer 14 \\
  --top-hidden-count 16 \\
  --top-ks 8,16,32,64,128 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-prompt-runs {max_prompt_runs}
"""


def _fetch(sftp, pack: str, mode: str, tag: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(pack, mode, tag)
    for name in [f"{prefix}_raw.csv", f"{prefix}_run.json"]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def _status_command(pack: str, mode: str, tag: str) -> str:
    prefix = _prefix(pack, mode, tag)
    return f"""
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'run_stage4_qwen_hidden_to_plt_mediation|{prefix}' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{prefix}* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*hidden_to_plt* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-038 Qwen hidden-to-PLT mediation remotely.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    max_prompt_runs = 6 if args.mode == "smoke" else 0

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.pack, args.mode, args.tag), get_pty=True)
        print(stdout.read().decode("utf-8", errors="replace"), end="")
        print(stderr.read().decode("utf-8", errors="replace"), end="")
        sftp.close()
        client.close()
        return 0
    if args.fetch_only:
        _fetch(sftp, args.pack, args.mode, args.tag)
        sftp.close()
        client.close()
        return 0

    for script in [
        "run_stage4_qwen_hidden_to_plt_mediation.py",
        "run_stage4_qwen_causal_cutter_validation.py",
        "run_stage4_qwen_evidence_linked_cutter_v2.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
    ]:
        _upload_research_script(base, sftp, script)
    _upload_prompt_runs(base, sftp, RUNS[args.pack], f"paperpack72_{args.pack}_prompt_runs.csv")
    base._mkdir_p(sftp, f"{REMOTE_STAGE}/logs")
    prefix = _prefix(args.pack, args.mode, args.tag)
    remote_script = f"{REMOTE_STAGE}/run_{prefix}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.pack, args.mode, args.tag, max_prompt_runs).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        log_path = f"{REMOTE_STAGE}/logs/{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        command = f"nohup bash {remote_script} > {log_path} 2>&1 & echo $!"
        _stdin, stdout, stderr = client.exec_command(command)
        pid = stdout.read().decode().strip()
        err = stderr.read().decode()
        print(json.dumps({"detached_remote_pid": pid, "remote_log": log_path, "remote_script": remote_script}, indent=2), flush=True)
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
    print(f"\nREMOTE_EXIT_STATUS={exit_status}", flush=True)
    sftp = client.open_sftp()
    _fetch(sftp, args.pack, args.mode, args.tag)
    sftp.close()
    client.close()
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
