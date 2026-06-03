#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"E:\Bridging")
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage2f_cross_model"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage2" / "cross_model"


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
    sftp = client.open_sftp()
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    files = [
        "stage2f_qwen_clt_feature_readout_smoke.json",
        "stage2f_llava_asset_format_smoke.json",
    ]
    for name in files:
        remote = f"{REMOTE_STAGE}/{name}"
        local = LOCAL_STAGE / name
        sftp.get(remote, str(local))
        print(f"fetched {remote} -> {local}")
    sftp.close()
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
