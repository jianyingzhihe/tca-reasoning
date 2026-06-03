#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage2f_cross_model"


def _load_connection() -> tuple[str, int, str]:
    text = (ROOT / "doc" / "5.9" / "server_connection_and_key_paths_2026-05-09.md").read_text(
        encoding="utf-8"
    )
    match = re.search(r"ssh -p (\d+) root@([^\s]+)", text)
    if not match:
        raise RuntimeError("Could not parse SSH host/port from connection doc.")
    password_match = re.search(r"Password:\s*```text\s*(.*?)\s*```", text, re.S)
    if not password_match:
        raise RuntimeError("Could not parse SSH password from connection doc.")
    return match.group(2), int(match.group(1)), password_match.group(1).strip()


def _remote_command() -> str:
    return r"""
set -e
cd /root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH=/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm:${PYTHONPATH:-}
mkdir -p /root/autodl-tmp/tca-reasoning/stage2f_cross_model
.venv/bin/python -m py_compile scripts/research/run_qwen_clt_feature_readout_smoke.py scripts/research/run_llava_asset_format_smoke.py
.venv/bin/python -u scripts/research/run_qwen_clt_feature_readout_smoke.py \
  --model-name /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --transcoder-set KokosDev/qwen2p5vl-7b-clt \
  --image-path /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg \
  --question "What does stop mean?" \
  --layers 0,13,26 \
  --hidden-state-offset 0 \
  --top-k 12 \
  --min-gpu-free-gb 18 \
  --out-json /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_qwen_clt_feature_readout_smoke.json
.venv/bin/python -u scripts/research/run_llava_asset_format_smoke.py \
  --repo-id KokosDev/llava15-7b-clt \
  --layer-index 0 \
  --download-sample \
  --out-json /root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_llava_asset_format_smoke.json
"""


def main() -> int:
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = _load_connection()
    local_files = [
        ROOT
        / "vlm-circuit-tracing"
        / "circuit_tracer_vlm"
        / "scripts"
        / "research"
        / "run_qwen_clt_feature_readout_smoke.py",
        ROOT
        / "vlm-circuit-tracing"
        / "circuit_tracer_vlm"
        / "scripts"
        / "research"
        / "run_llava_asset_format_smoke.py",
    ]

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
    for local in local_files:
        remote = f"{REMOTE_ROOT}/scripts/research/{local.name}"
        sftp.put(str(local), remote)
        sftp.chmod(remote, 0o755)
        print(f"uploaded {local.name} -> {remote}")
    sftp.close()

    remote_script = f"{REMOTE_STAGE}/run_stage2f_cross_model.sh"
    sftp = client.open_sftp()
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_command().replace("\r\n", "\n").lstrip())
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    command = f"bash {remote_script}"
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
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
        if time.time() - start > 3600:
            stdout.channel.close()
            raise TimeoutError("remote command exceeded 1 hour")

    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")

    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")
    client.close()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
