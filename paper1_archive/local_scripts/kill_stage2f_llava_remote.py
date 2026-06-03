#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"E:\Bridging")


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
    cmd = (
        "pkill -f run_llava_base_hook_forward_smoke.py || true; "
        "sleep 2; "
        "ps -eo pid,etime,cmd | grep run_llava_base_hook_forward_smoke | grep -v grep || true"
    )
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace")
    if err:
        print("STDERR:")
        print(err)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
