#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage2f_cross_model"
REMOTE_ASSETS = f"{REMOTE_STAGE}/stage2g_assets"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage2" / "cross_model"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    base = _load_base_runner()
    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20)
    sftp = client.open_sftp()
    base._upload_scripts(sftp)
    local_restoration = (
        ROOT
        / "vlm-circuit-tracing"
        / "circuit_tracer_vlm"
        / "scripts"
        / "research"
        / "run_cross_model_feature_restoration_smoke.py"
    )
    remote_restoration = f"{REMOTE_ROOT}/scripts/research/run_cross_model_feature_restoration_smoke.py"
    base._put_file(sftp, local_restoration, remote_restoration)
    sftp.chmod(remote_restoration, 0o755)
    print("uploaded run_cross_model_feature_restoration_smoke.py")
    base._upload_assets(sftp)
    remote_script = f"{REMOTE_STAGE}/run_stage2g_feature_restoration.sh"
    script = f"""#!/usr/bin/env bash
set -e
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub
STAGE={REMOTE_STAGE}
ASSET_ROOT={REMOTE_ASSETS}
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* | head -n 1)
LLAVA_MODEL_DIR=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf

echo '--- Stage 2G feature restoration disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
.venv/bin/python -m py_compile scripts/research/run_cross_model_feature_intervention_smoke.py
.venv/bin/python -m py_compile scripts/research/run_cross_model_feature_restoration_smoke.py

echo '--- Qwen mask-to-clean feature restoration smoke ---'
.venv/bin/python -u scripts/research/run_cross_model_feature_restoration_smoke.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --annotation-roots "$ASSET_ROOT,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_core24_easy,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_mainline16" \\
  --work-dir "$STAGE/stage2g_qwen_restoration_work" \\
  --samples okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3658865 \\
  --prompts B_direct,D_visual_only \\
  --layer 26 \\
  --top-k-features 4 \\
  --scales 0.5,1.0,1.5 \\
  --out-json "$STAGE/stage2g_qwen_feature_restoration_smoke.json" \\
  --out-csv "$STAGE/stage2g_qwen_feature_restoration_smoke.csv"

echo '--- LLaVA mask-to-clean feature restoration smoke ---'
.venv/bin/python -u scripts/research/run_cross_model_feature_restoration_smoke.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --transcoder-ref KokosDev/llava15-7b-clt \\
  --annotation-roots "$ASSET_ROOT,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_core24_easy,/root/autodl-tmp/tca-reasoning/annotation/okvqa_evidence_labelme_round4_mainline16" \\
  --work-dir "$STAGE/stage2g_llava_restoration_work" \\
  --samples okvqa_val_2847255,okvqa_val_4157235,okvqa_val_3658865 \\
  --prompts B_direct,D_visual_only \\
  --layer 15 \\
  --top-k-features 4 \\
  --scales 0.5,1.0,1.5 \\
  --out-json "$STAGE/stage2g_llava_feature_restoration_smoke.json" \\
  --out-csv "$STAGE/stage2g_llava_feature_restoration_smoke.csv"
"""
    with sftp.file(remote_script, "w") as handle:
        handle.write(script)
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
            raise TimeoutError("Stage 2G feature restoration exceeded 3 hours")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={status}")
    sftp = client.open_sftp()
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    for name in [
        "stage2g_qwen_feature_restoration_smoke.json",
        "stage2g_qwen_feature_restoration_smoke.csv",
        "stage2g_llava_feature_restoration_smoke.json",
        "stage2g_llava_feature_restoration_smoke.csv",
    ]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_STAGE / name))
            print(f"fetched {name}")
        except FileNotFoundError:
            print(f"missing {name}")
    sftp.close()
    client.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
