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
    for name in [
        "run_qwen_token_position_mapping_smoke.py",
        "run_qwen_clean_vs_mask_feature_readout.py",
    ]:
        local = (
            ROOT
            / "vlm-circuit-tracing"
            / "circuit_tracer_vlm"
            / "scripts"
            / "research"
            / name
        )
        remote = f"{REMOTE_ROOT}/scripts/research/{name}"
        sftp.put(str(local), remote)
        sftp.chmod(remote, 0o755)
        print(f"uploaded {name}")
    remote_script = f"{REMOTE_STAGE}/rerun_stage2f3_qwen.sh"
    cmd = f"""
set -e
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
.venv/bin/python -m py_compile scripts/research/run_qwen_token_position_mapping_smoke.py scripts/research/run_qwen_clean_vs_mask_feature_readout.py
QWEN_MODEL=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
STAGE={REMOTE_STAGE}
ASSET_ROOT=$STAGE/qwen_q3_assets
.venv/bin/python -u scripts/research/run_qwen_token_position_mapping_smoke.py --model-name "$QWEN_MODEL" --transcoder-set KokosDev/qwen2p5vl-7b-clt --image-path /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg --question "What does stop mean?" --layers 0,13,26 --top-k 12 --out-json "$STAGE/stage2f_qwen_position_mapping_smoke_192716.json" --out-token-csv "$STAGE/stage2f_qwen_position_mapping_smoke_192716_tokens.csv" --out-bucket-csv "$STAGE/stage2f_qwen_position_mapping_smoke_192716_buckets.csv"
.venv/bin/python -u scripts/research/run_qwen_token_position_mapping_smoke.py --model-name "$QWEN_MODEL" --transcoder-set KokosDev/qwen2p5vl-7b-clt --image-path "$ASSET_ROOT/images/COCO_val2014_000000284725.jpg" --question "What country might this be based on the writing on the bus? Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>." --layers 0,13,26 --top-k 12 --out-json "$STAGE/stage2f_qwen_position_mapping_2847255.json" --out-token-csv "$STAGE/stage2f_qwen_position_mapping_2847255_tokens.csv" --out-bucket-csv "$STAGE/stage2f_qwen_position_mapping_2847255_buckets.csv"
.venv/bin/python -u scripts/research/run_qwen_clean_vs_mask_feature_readout.py --model-name "$QWEN_MODEL" --transcoder-set KokosDev/qwen2p5vl-7b-clt --annotation-roots "$ASSET_ROOT,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_core24_easy,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_mainline16" --work-dir "$STAGE/qwen_q3_work" --samples okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3605295 --prompts B_direct,D_visual_only --layers 0,13,26 --top-k 20 --out-json "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout.json" --out-csv "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout.csv" --out-summary-csv "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout_summary.csv"
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
        if time.time() - start > 3600:
            stdout.channel.close()
            raise TimeoutError("rerun qwen exceeded 1 hour")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")
    client.close()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
