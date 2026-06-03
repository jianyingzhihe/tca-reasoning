#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage3_cross_model"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage3" / "cross_model"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_stage3_runner():
    spec = importlib.util.spec_from_file_location(
        "stage3_runner", ROOT / "scripts" / "local" / "run_stage3_cross_model_dual_track_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _remote_script(mode: str, max_new_tokens: int) -> str:
    suffix = "_smoke" if mode == "smoke" else ""
    max_pairs = 3 if mode == "smoke" else 0
    return f"""#!/usr/bin/env bash
set -e
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub
STAGE={REMOTE_STAGE}
ASSET_ROOT={REMOTE_ASSETS}
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* | head -n 1 || true)

echo '--- Stage3-13 Qwen generation bridge v2 disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  scripts/research/run_cross_model_wrong_target_negative_control_smoke.py \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage3_qwen_multifeature_sequence_bridge.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: generation bridge v2 skipped'
  exit 0
fi

echo '--- Stage3-13 Qwen2.5-VL PLT generation bridge v2 ---'
.venv/bin/python -u scripts/research/run_stage3_qwen_multifeature_sequence_bridge.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --asset-id qwen2p5vl_plt \\
  --annotation-roots "$ASSET_ROOT" \\
  --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
  --bridge-manifest "$STAGE/stage3_qwen_generation_bridge_v2_manifest.csv" \\
  --topks 1,4,8,16,32 \\
  --max-pairs {max_pairs} \\
  --max-new-tokens {max_new_tokens} \\
  --out-json "$STAGE/stage3_qwen2p5vl_plt_generation_bridge_v2{suffix}.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_plt_generation_bridge_v2{suffix}.csv"

echo '--- Stage3-13 Qwen2.5-VL CLT generation bridge v2 ---'
.venv/bin/python -u scripts/research/run_stage3_qwen_multifeature_sequence_bridge.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --asset-id qwen2p5vl_clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
  --bridge-manifest "$STAGE/stage3_qwen_generation_bridge_v2_manifest.csv" \\
  --topks 1,4,8,16,32 \\
  --max-pairs {max_pairs} \\
  --max-new-tokens {max_new_tokens} \\
  --out-json "$STAGE/stage3_qwen2p5vl_clt_generation_bridge_v2{suffix}.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_clt_generation_bridge_v2{suffix}.csv"
"""


def _fetch_outputs(sftp, mode: str) -> None:
    suffix = "_smoke" if mode == "smoke" else ""
    for name in [
        f"stage3_qwen2p5vl_plt_generation_bridge_v2{suffix}.json",
        f"stage3_qwen2p5vl_plt_generation_bridge_v2{suffix}.csv",
        f"stage3_qwen2p5vl_clt_generation_bridge_v2{suffix}.json",
        f"stage3_qwen2p5vl_clt_generation_bridge_v2{suffix}.csv",
    ]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_STAGE / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage3-13 Qwen generation bridge v2 on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    args = parser.parse_args()

    base = _load_base_runner()
    stage3 = _load_stage3_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    for script_name in [
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_cross_model_wrong_target_negative_control_smoke.py",
        "run_stage2o_attribution_weighted_feature_bridge.py",
        "run_stage3_qwen_multifeature_sequence_bridge.py",
    ]:
        stage3._upload_research_script(base, sftp, script_name)
    stage3._upload_stage3_assets(base, sftp)
    base._put_file(
        sftp,
        LOCAL_STAGE / "stage3_qwen_generation_bridge_v2_manifest.csv",
        f"{REMOTE_STAGE}/stage3_qwen_generation_bridge_v2_manifest.csv",
    )
    remote_script = f"{REMOTE_STAGE}/run_stage3_qwen_generation_bridge_v2_{args.mode}.sh"
    base._mkdir_p(sftp, REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.mode, args.max_new_tokens).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    _stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
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
        if time.time() - start > args.timeout_seconds:
            stdout.channel.close()
            raise TimeoutError("remote Stage3-13 Qwen generation bridge v2 exceeded timeout")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")

    sftp = client.open_sftp()
    _fetch_outputs(sftp, args.mode)
    sftp.close()
    client.close()
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
