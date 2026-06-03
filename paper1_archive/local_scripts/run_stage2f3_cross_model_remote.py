#!/usr/bin/env python3
from __future__ import annotations

import os
import posixpath
import re
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage2f_cross_model"
REMOTE_ASSETS = f"{REMOTE_STAGE}/qwen_q3_assets"


def _load_connection() -> tuple[str, int, str]:
    text = (ROOT / "doc" / "5.9" / "server_connection_and_key_paths_2026-05-09.md").read_text(
        encoding="utf-8"
    )
    match = re.search(r"ssh -p (\d+) root@([^\s]+)", text)
    password_match = re.search(r"Password:\s*```text\s*(.*?)\s*```", text, re.S)
    if not match or not password_match:
        raise RuntimeError("Could not parse server connection details.")
    return match.group(2), int(match.group(1)), password_match.group(1).strip()


def _mkdir_p(sftp, remote_dir: str) -> None:
    parts = []
    current = remote_dir
    while current not in {"", "/"}:
        parts.append(current)
        current = posixpath.dirname(current)
    for path in reversed(parts):
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def _put_file(sftp, local: Path, remote: str) -> None:
    _mkdir_p(sftp, posixpath.dirname(remote))
    sftp.put(str(local), remote)


def _upload_assets(sftp) -> None:
    images = [
        "COCO_val2014_000000284725.jpg",
        "COCO_val2014_000000415723.jpg",
        "COCO_val2014_000000360529.jpg",
    ]
    mainline_images = ROOT / "annotation" / "okvqa_evidence_labelme_round4_mainline16" / "images"
    core_masks = ROOT / "annotation" / "okvqa_evidence_labelme_round4_core24_easy" / "exported_masks"
    for image_name in images:
        local_image = mainline_images / image_name
        if local_image.exists():
            _put_file(sftp, local_image, f"{REMOTE_ASSETS}/images/{image_name}")
        stem = Path(image_name).stem
        for mask_name in ["answer.png", "relate.png"]:
            local_mask = core_masks / stem / mask_name
            if local_mask.exists():
                _put_file(sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}")


def _remote_script() -> str:
    return r"""
set -e
cd /root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH=/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm:${PYTHONPATH:-}
mkdir -p /root/autodl-tmp/tca-reasoning/stage2f_cross_model

.venv/bin/python -m py_compile \
  scripts/research/run_qwen_token_position_mapping_smoke.py \
  scripts/research/run_qwen_clean_vs_mask_feature_readout.py \
  scripts/research/run_llava_base_hook_forward_smoke.py

QWEN_MODEL=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
STAGE=/root/autodl-tmp/tca-reasoning/stage2f_cross_model
ASSET_ROOT=$STAGE/qwen_q3_assets

.venv/bin/python -u scripts/research/run_qwen_token_position_mapping_smoke.py \
  --model-name "$QWEN_MODEL" \
  --transcoder-set KokosDev/qwen2p5vl-7b-clt \
  --image-path /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg \
  --question "What does stop mean?" \
  --layers 0,13,26 \
  --top-k 12 \
  --out-json "$STAGE/stage2f_qwen_position_mapping_smoke_192716.json" \
  --out-token-csv "$STAGE/stage2f_qwen_position_mapping_smoke_192716_tokens.csv" \
  --out-bucket-csv "$STAGE/stage2f_qwen_position_mapping_smoke_192716_buckets.csv"

.venv/bin/python -u scripts/research/run_qwen_token_position_mapping_smoke.py \
  --model-name "$QWEN_MODEL" \
  --transcoder-set KokosDev/qwen2p5vl-7b-clt \
  --image-path "$ASSET_ROOT/images/COCO_val2014_000000284725.jpg" \
  --question "What country might this be based on the writing on the bus? Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>." \
  --layers 0,13,26 \
  --top-k 12 \
  --out-json "$STAGE/stage2f_qwen_position_mapping_2847255.json" \
  --out-token-csv "$STAGE/stage2f_qwen_position_mapping_2847255_tokens.csv" \
  --out-bucket-csv "$STAGE/stage2f_qwen_position_mapping_2847255_buckets.csv"

.venv/bin/python -u scripts/research/run_qwen_clean_vs_mask_feature_readout.py \
  --model-name "$QWEN_MODEL" \
  --transcoder-set KokosDev/qwen2p5vl-7b-clt \
  --annotation-roots "$ASSET_ROOT,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_core24_easy,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_mainline16" \
  --work-dir "$STAGE/qwen_q3_work" \
  --samples okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3605295 \
  --prompts B_direct,D_visual_only \
  --layers 0,13,26 \
  --top-k 20 \
  --out-json "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout.json" \
  --out-csv "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout.csv" \
  --out-summary-csv "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout_summary.csv"

.venv/bin/python -u scripts/research/run_llava_base_hook_forward_smoke.py \
  --model-name llava-hf/llava-1.5-7b-hf \
  --image-path /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg \
  --question "What does stop mean?" \
  --out-json "$STAGE/stage2f_llava_base_hook_forward_smoke.json" \
  --disk-path /root/autodl-tmp \
  --min-free-gb 40 \
  --min-gpu-free-gb 18 \
  --layer-index 0
"""


def main() -> int:
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = _load_connection()
    scripts = [
        ROOT
        / "vlm-circuit-tracing"
        / "circuit_tracer_vlm"
        / "scripts"
        / "research"
        / "run_qwen_token_position_mapping_smoke.py",
        ROOT
        / "vlm-circuit-tracing"
        / "circuit_tracer_vlm"
        / "scripts"
        / "research"
        / "run_qwen_clean_vs_mask_feature_readout.py",
        ROOT
        / "vlm-circuit-tracing"
        / "circuit_tracer_vlm"
        / "scripts"
        / "research"
        / "run_llava_base_hook_forward_smoke.py",
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
    for local in scripts:
        remote = f"{REMOTE_ROOT}/scripts/research/{local.name}"
        _put_file(sftp, local, remote)
        sftp.chmod(remote, 0o755)
        print(f"uploaded {local.name} -> {remote}")
    _upload_assets(sftp)
    print(f"uploaded qwen q3 assets -> {REMOTE_ASSETS}")
    remote_script = f"{REMOTE_STAGE}/run_stage2f3_cross_model.sh"
    _mkdir_p(sftp, REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script().replace("\r\n", "\n").lstrip())
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
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
        if time.time() - start > 7200:
            stdout.channel.close()
            raise TimeoutError("remote Stage 2F-3 command exceeded 2 hours")
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
