#!/usr/bin/env python3
from __future__ import annotations

import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage2f_cross_model"


def main() -> int:
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    text = (ROOT / "doc" / "5.9" / "server_connection_and_key_paths_2026-05-09.md").read_text(
        encoding="utf-8"
    )
    match = re.search(r"ssh -p (\d+) root@([^\s]+)", text)
    password_match = re.search(r"Password:\s*```text\s*(.*?)\s*```", text, re.S)
    if not match or not password_match:
        raise RuntimeError("Could not parse connection details.")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=match.group(2),
        port=int(match.group(1)),
        username="root",
        password=password_match.group(1).strip(),
        timeout=20,
    )
    sftp = client.open_sftp()
    local = (
        ROOT
        / "vlm-circuit-tracing"
        / "circuit_tracer_vlm"
        / "scripts"
        / "research"
        / "run_llava_base_hook_forward_smoke.py"
    )
    remote = f"{REMOTE_ROOT}/scripts/research/run_llava_base_hook_forward_smoke.py"
    sftp.put(str(local), remote)
    sftp.chmod(remote, 0o755)
    remote_script = f"{REMOTE_STAGE}/run_stage2f3_llava_skip.sh"
    cmd = f"""
set -e
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
.venv/bin/python -m py_compile scripts/research/run_llava_base_hook_forward_smoke.py
.venv/bin/python -u scripts/research/run_llava_base_hook_forward_smoke.py \
  --model-name llava-hf/llava-1.5-7b-hf \
  --image-path /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg \
  --question "What does stop mean?" \
  --out-json {REMOTE_STAGE}/stage2f_llava_base_hook_forward_smoke.json \
  --disk-path /root/autodl-tmp \
  --skip-base-forward
"""
    with sftp.file(remote_script, "w") as handle:
        handle.write(cmd.replace("\r\n", "\n").lstrip())
    sftp.chmod(remote_script, 0o755)
    sftp.close()
    stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
    stdout.channel.settimeout(0.0)
    start = time.time()
    while not stdout.channel.exit_status_ready():
        try:
            data = stdout.channel.recv(8192)
            if data:
                print(data.decode("utf-8", errors="replace"), end="")
        except socket.timeout:
            pass
        time.sleep(0.5)
        if time.time() - start > 300:
            stdout.channel.close()
            raise TimeoutError("llava skip remote command exceeded 5 minutes")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")
    client.close()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
