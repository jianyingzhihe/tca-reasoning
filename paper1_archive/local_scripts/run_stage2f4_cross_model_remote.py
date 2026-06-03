#!/usr/bin/env python3
from __future__ import annotations

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


def _upload_research_scripts(sftp) -> None:
    names = [
        "run_qwen_token_position_mapping_smoke.py",
        "run_qwen_clean_vs_mask_feature_readout.py",
        "run_llava_base_hook_forward_smoke.py",
    ]
    for name in names:
        local = (
            ROOT
            / "vlm-circuit-tracing"
            / "circuit_tracer_vlm"
            / "scripts"
            / "research"
            / name
        )
        remote = f"{REMOTE_ROOT}/scripts/research/{name}"
        _put_file(sftp, local, remote)
        sftp.chmod(remote, 0o755)
        print(f"uploaded {name} -> {remote}")


def _upload_qwen_assets(sftp) -> None:
    image_roots = [
        ROOT / "annotation" / "okvqa_evidence_labelme_round4_core24_easy" / "images",
        ROOT / "annotation" / "okvqa_evidence_labelme_round4_mainline16" / "images",
    ]
    mask_roots = [
        ROOT / "annotation" / "okvqa_evidence_labelme_round4_core24_easy" / "exported_masks",
        ROOT / "annotation" / "okvqa_evidence_labelme_round4_mainline16" / "exported_masks",
    ]
    image_names = [
        "COCO_val2014_000000284725.jpg",
        "COCO_val2014_000000365886.jpg",
        "COCO_val2014_000000415723.jpg",
    ]
    for image_name in image_names:
        local_image = next((root / image_name for root in image_roots if (root / image_name).exists()), None)
        if local_image is not None:
            _put_file(sftp, local_image, f"{REMOTE_ASSETS}/images/{image_name}")
            print(f"uploaded image {image_name}")
        stem = Path(image_name).stem
        for mask_name in ["answer.png", "relate.png"]:
            local_mask = next(
                (root / stem / mask_name for root in mask_roots if (root / stem / mask_name).exists()),
                None,
            )
            if local_mask is not None:
                _put_file(sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}")
                print(f"uploaded mask {stem}/{mask_name}")


def _remote_script() -> str:
    return r"""
set -e
cd /root/autodl-tmp/tca-reasoning/circuit_tracer_vlm
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH=/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm:${PYTHONPATH:-}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub
mkdir -p /root/autodl-tmp/tca-reasoning/stage2f_cross_model/logs

echo '--- stage2f4 disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

.venv/bin/python -m py_compile \
  scripts/research/run_qwen_token_position_mapping_smoke.py \
  scripts/research/run_qwen_clean_vs_mask_feature_readout.py \
  scripts/research/run_llava_base_hook_forward_smoke.py

QWEN_MODEL=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
STAGE=/root/autodl-tmp/tca-reasoning/stage2f_cross_model
ASSET_ROOT=$STAGE/qwen_q3_assets

echo '--- qwen 3658865 position mapping ---'
.venv/bin/python -u scripts/research/run_qwen_token_position_mapping_smoke.py \
  --model-name "$QWEN_MODEL" \
  --transcoder-set KokosDev/qwen2p5vl-7b-clt \
  --image-path "$ASSET_ROOT/images/COCO_val2014_000000365886.jpg" \
  --question "What brand of phone is this? Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>." \
  --layers 0,13,26 \
  --top-k 12 \
  --out-json "$STAGE/stage2f_qwen_position_mapping_3658865.json" \
  --out-token-csv "$STAGE/stage2f_qwen_position_mapping_3658865_tokens.csv" \
  --out-bucket-csv "$STAGE/stage2f_qwen_position_mapping_3658865_buckets.csv"

echo '--- qwen q3 3-case rerun ---'
.venv/bin/python -u scripts/research/run_qwen_clean_vs_mask_feature_readout.py \
  --model-name "$QWEN_MODEL" \
  --transcoder-set KokosDev/qwen2p5vl-7b-clt \
  --annotation-roots "$ASSET_ROOT,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_core24_easy,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_mainline16" \
  --work-dir "$STAGE/qwen_q3_work_stage2f4" \
  --samples okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3658865 \
  --prompts B_direct,D_visual_only \
  --layers 0,13,26 \
  --top-k 20 \
  --out-json "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout_3case.json" \
  --out-csv "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout_3case.csv" \
  --out-summary-csv "$STAGE/stage2f_qwen_clean_vs_mask_feature_readout_3case_summary.csv"

echo '--- llava modelscope download attempt ---'
.venv/bin/python - <<'PY'
import json
import shutil
import time
from pathlib import Path

out = Path('/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_llava_modelscope_download.json')
payload = {
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'candidate_repos': ['swift/llava-1.5-7b-hf'],
    'cache_dir': '/root/autodl-tmp/tca-reasoning/data/modelscope_cache',
    'disk_before': {},
}
usage = shutil.disk_usage('/root/autodl-tmp')
payload['disk_before'] = {
    'total_gb': round(usage.total / (1024 ** 3), 3),
    'used_gb': round(usage.used / (1024 ** 3), 3),
    'free_gb': round(usage.free / (1024 ** 3), 3),
}
try:
    from modelscope.hub.snapshot_download import snapshot_download
    model_dir = snapshot_download(
        'swift/llava-1.5-7b-hf',
        cache_dir='/root/autodl-tmp/tca-reasoning/data/modelscope_cache',
    )
    payload['status'] = 'pass'
    payload['model_dir'] = model_dir
except Exception as exc:
    payload['status'] = 'failed'
    payload['error_type'] = type(exc).__name__
    payload['error'] = str(exc)[:4000]
usage = shutil.disk_usage('/root/autodl-tmp')
payload['disk_after'] = {
    'total_gb': round(usage.total / (1024 ** 3), 3),
    'used_gb': round(usage.used / (1024 ** 3), 3),
    'free_gb': round(usage.free / (1024 ** 3), 3),
}
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(payload, ensure_ascii=False))
PY

LLAVA_MODEL_DIR=$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/root/autodl-tmp/tca-reasoning/stage2f_cross_model/stage2f_llava_modelscope_download.json')
d = json.loads(p.read_text(encoding='utf-8'))
print(d.get('model_dir', ''))
PY
)

if [ -n "$LLAVA_MODEL_DIR" ] && [ -d "$LLAVA_MODEL_DIR" ]; then
  echo "--- llava hook-forward from ModelScope dir: $LLAVA_MODEL_DIR ---"
  .venv/bin/python -u scripts/research/run_llava_base_hook_forward_smoke.py \
    --model-name "$LLAVA_MODEL_DIR" \
    --image-path /root/autodl-tmp/tca-reasoning/data/okvqa/images/val2014/COCO_val2014_000000192716.jpg \
    --question "What does stop mean?" \
    --out-json "$STAGE/stage2f_llava_base_hook_forward_smoke_modelscope.json" \
    --disk-path /root/autodl-tmp \
    --min-free-gb 20 \
    --min-gpu-free-gb 18 \
    --layer-index 0
else
  echo 'LLaVA ModelScope download failed or returned no local directory; skipping hook-forward.'
fi
"""


def main() -> int:
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = _load_connection()
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
    _upload_research_scripts(sftp)
    _upload_qwen_assets(sftp)
    remote_script = f"{REMOTE_STAGE}/run_stage2f4_cross_model.sh"
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
        if time.time() - start > 10800:
            stdout.channel.close()
            raise TimeoutError("remote Stage 2F-4 command exceeded 3 hours")
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
