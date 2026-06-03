#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
import time
from pathlib import Path

import paramiko


ROOT = Path(r"E:\Bridging")
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage6_defensive"
REMOTE_PYTHON = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm/.venv/bin/python"
LOCAL_STAGE4_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
LOCAL_STAGE6_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
ANALYZER = ROOT / "scripts" / "local" / "analyze_stage6_qwen_grouped_composition.py"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner",
        ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _remote_script(mode: str, tag: str, max_samples: int, max_prompts: int) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p {REMOTE_STAGE}/outputs {REMOTE_STAGE}/logs
cd {REMOTE_STAGE}
PY="{REMOTE_PYTHON}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || true)"
fi
if [ -z "$PY" ]; then
  echo "blocked: no python interpreter found" >&2
  exit 2
fi
echo '--- Stage6-020 Qwen grouped composition smoke ---'
date
hostname || true
nvidia-smi || true
df -h /root/autodl-tmp || true
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
export STAGE6_DEFENSIVE_ROOT="{REMOTE_STAGE}"
export STAGE6_DEFENSIVE_OUT_DIR="{REMOTE_STAGE}/outputs"
"$PY" -m py_compile analyze_stage6_qwen_grouped_composition.py
"$PY" analyze_stage6_qwen_grouped_composition.py \\
  --mode {mode} \\
  --tag {tag} \\
  --max-samples {max_samples} \\
  --max-prompts {max_prompts}
echo '--- outputs ---'
ls -lh outputs || true
date
"""


def _fetch(sftp, mode: str, tag: str) -> None:
    stem = f"stage6_defensive_qwen_grouped_composition_{mode}_{tag}"
    for suffix in [
        "_topk_summary.csv",
        "_layer_band_summary.csv",
        "_topk_nonmonotonic.csv",
        "_decision.json",
    ]:
        name = stem + suffix
        remote = f"{REMOTE_STAGE}/outputs/{name}"
        local = LOCAL_STAGE6_CROSS / name
        try:
            sftp.get(remote, str(local))
            print(f"fetched {name}", flush=True)
        except OSError as exc:
            print(f"missing remote output {remote}: {exc}", flush=True)


def _status(client) -> None:
    cmd = f"""
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'stage6_defensive|analyze_stage6_qwen_grouped_composition' | grep -v grep || true
ls -lh {REMOTE_STAGE}/outputs 2>/dev/null || true
ls -lh {REMOTE_STAGE}/logs 2>/dev/null || true
tail -n 80 {REMOTE_STAGE}/logs/qwen_grouped_composition_*.log 2>/dev/null || true
nvidia-smi || true
df -h /root/autodl-tmp || true
"""
    _stdin, stdout, stderr = client.exec_command(cmd)
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace")
    if err:
        print(err, file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage6-020 Qwen grouped composition diagnostic on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="defensive_v1")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()

    base = _load_base_runner()
    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)

    if args.status:
        _status(client)
        client.close()
        return 0

    sftp = client.open_sftp()
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_STAGE}/outputs")
    base._mkdir_p(sftp, f"{REMOTE_STAGE}/logs")

    if args.fetch_only:
        _fetch(sftp, args.mode, args.tag)
        sftp.close()
        client.close()
        return 0

    remote_analyzer = f"{REMOTE_STAGE}/analyze_stage6_qwen_grouped_composition.py"
    base._put_file(sftp, ANALYZER, remote_analyzer)

    input_files = [
        LOCAL_STAGE4_CROSS / "stage4_qwen_feature_route_featureroute_v1_route_metrics.csv",
        LOCAL_STAGE4_CROSS / "stage4_qwen_route_first_routefirst_v1_route_candidates.csv",
    ]
    # Mirror the local analyzer's expected repository-relative paths inside the defensive stage.
    base._mkdir_p(sftp, f"{REMOTE_STAGE}/doc/experiments/stage4/cross_model")
    base._mkdir_p(sftp, f"{REMOTE_STAGE}/doc/experiments/stage6/cross_model")
    for path in input_files:
        base._put_file(sftp, path, f"{REMOTE_STAGE}/doc/experiments/stage4/cross_model/{path.name}")

    remote_script = f"{REMOTE_STAGE}/run_qwen_grouped_composition_{args.mode}_{args.tag}.sh"
    script = _remote_script(args.mode, args.tag, args.max_samples, args.max_prompts)
    with sftp.file(remote_script, "w") as handle:
        handle.write(script.replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log = f"{REMOTE_STAGE}/logs/qwen_grouped_composition_{args.mode}_{stamp}.log"
        cmd = f"nohup bash {remote_script} > {remote_log} 2>&1 < /dev/null & echo $!"
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
    _fetch(sftp, args.mode, args.tag)
    sftp.close()
    client.close()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
