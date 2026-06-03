#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"E:\Bridging")
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage2f_cross_model"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage2" / "cross_model"


FILES = [
    "stage2f_qwen_position_mapping_smoke_192716.json",
    "stage2f_qwen_position_mapping_smoke_192716_tokens.csv",
    "stage2f_qwen_position_mapping_smoke_192716_buckets.csv",
    "stage2f_qwen_position_mapping_2847255.json",
    "stage2f_qwen_position_mapping_2847255_tokens.csv",
    "stage2f_qwen_position_mapping_2847255_buckets.csv",
    "stage2f_qwen_clean_vs_mask_feature_readout.json",
    "stage2f_qwen_clean_vs_mask_feature_readout.csv",
    "stage2f_qwen_clean_vs_mask_feature_readout_summary.csv",
    "stage2f_llava_base_hook_forward_smoke.json",
]


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
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        remote = f"{REMOTE_STAGE}/{name}"
        local = LOCAL_STAGE / name
        try:
            sftp.get(remote, str(local))
            print(f"fetched {name}")
        except FileNotFoundError:
            print(f"missing {name}")
    sftp.close()
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
