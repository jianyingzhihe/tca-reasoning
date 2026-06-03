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


def _remote_script(max_new_tokens: int, *, pair_manifest_name: str, output_suffix: str) -> str:
    plt_json = f"stage3_qwen2p5vl_plt_decoded_bridge{output_suffix}.json"
    plt_csv = f"stage3_qwen2p5vl_plt_decoded_bridge{output_suffix}.csv"
    clt_json = f"stage3_qwen2p5vl_clt_decoded_bridge{output_suffix}.json"
    clt_csv = f"stage3_qwen2p5vl_clt_decoded_bridge{output_suffix}.csv"
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

echo '--- Stage3 Qwen decoded bridge disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  scripts/research/run_stage3_qwen_feature_decode_bridge.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: decoded bridge skipped'
  exit 0
fi

echo '--- Stage3 Qwen2.5-VL PLT decoded feature bridge ---'
.venv/bin/python -u scripts/research/run_stage3_qwen_feature_decode_bridge.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --asset-id qwen2p5vl_plt \\
  --annotation-roots "$ASSET_ROOT" \\
  --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
  --pair-manifest "$STAGE/{pair_manifest_name}" \\
  --max-new-tokens {max_new_tokens} \\
  --out-json "$STAGE/{plt_json}" \\
  --out-csv "$STAGE/{plt_csv}"

echo '--- Stage3 Qwen2.5-VL CLT decoded feature bridge ---'
.venv/bin/python -u scripts/research/run_stage3_qwen_feature_decode_bridge.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --asset-id qwen2p5vl_clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
  --pair-manifest "$STAGE/{pair_manifest_name}" \\
  --max-new-tokens {max_new_tokens} \\
  --out-json "$STAGE/{clt_json}" \\
  --out-csv "$STAGE/{clt_csv}"
"""


def _fetch_outputs(sftp, *, output_suffix: str) -> None:
    for name in [
        f"stage3_qwen2p5vl_plt_decoded_bridge{output_suffix}.json",
        f"stage3_qwen2p5vl_plt_decoded_bridge{output_suffix}.csv",
        f"stage3_qwen2p5vl_clt_decoded_bridge{output_suffix}.json",
        f"stage3_qwen2p5vl_clt_decoded_bridge{output_suffix}.csv",
    ]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_STAGE / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage3 Qwen decoded bridge smoke on AutoDL.")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--rankaware", action="store_true", help="Use rank-aware decoded bridge manifest and output suffix.")
    args = parser.parse_args()
    pair_manifest_name = (
        "stage3_qwen_decoded_bridge_rankaware_manifest.csv"
        if args.rankaware
        else "stage3_qwen_decoded_bridge_manifest.csv"
    )
    output_suffix = "_rankaware" if args.rankaware else ""

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
        "run_stage2o_cross_model_source_control_probe.py",
        "run_stage3_qwen_feature_decode_bridge.py",
    ]:
        stage3._upload_research_script(base, sftp, script_name)
    stage3._upload_stage3_assets(base, sftp)
    base._put_file(sftp, LOCAL_STAGE / pair_manifest_name, f"{REMOTE_STAGE}/{pair_manifest_name}")
    remote_script = f"{REMOTE_STAGE}/run_stage3_qwen_decoded_bridge.sh"
    base._mkdir_p(sftp, REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                args.max_new_tokens,
                pair_manifest_name=pair_manifest_name,
                output_suffix=output_suffix,
            ).replace("\r\n", "\n")
        )
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
            raise TimeoutError("remote Stage3 Qwen decoded bridge exceeded timeout")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")

    sftp = client.open_sftp()
    _fetch_outputs(sftp, output_suffix=output_suffix)
    sftp.close()
    client.close()
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
