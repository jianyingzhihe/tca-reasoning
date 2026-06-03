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
    host = match.group(2)
    port = int(match.group(1))
    password = password_match.group(1).strip()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20)
    cmd = """
ps -eo pid,etime,cmd | grep -E 'run_qwen_clt_feature|run_llava_asset|Qwen2_5|python -u scripts/research' | grep -v grep || true
echo '--- stage dir ---'
ls -lh /root/autodl-tmp/tca-reasoning/stage2f_cross_model | tail -30
echo '--- json statuses ---'
python - <<'PY'
import json
from pathlib import Path
for p in Path('/root/autodl-tmp/tca-reasoning/stage2f_cross_model').glob('stage2f_*smoke.json'):
    try:
        data=json.loads(p.read_text())
        print(p.name, data.get('decision', {}))
    except Exception as exc:
        print(p.name, type(exc).__name__, str(exc)[:200])
PY
"""
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
