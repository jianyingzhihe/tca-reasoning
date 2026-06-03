#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"


def _load_paperpack_runner():
    spec = importlib.util.spec_from_file_location(
        "stage3_paperpack_runner", ROOT / "scripts" / "local" / "run_stage3_paperpack_plt_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _remote_script(runner, pack: str) -> str:
    suffix = f"{pack}_controls_full"
    return f"""#!/usr/bin/env bash
set -e
cd {runner.REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={runner.REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub
STAGE={runner.REMOTE_STAGE}
ASSET_ROOT={runner.REMOTE_ASSETS}
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* | head -n 1 || true)
SAMPLE_MANIFEST="$STAGE/paperpack_{pack}_manifest.csv"
RUN_MANIFEST="$STAGE/paperpack_{pack}_prompt_runs.csv"

echo '--- Stage3 Qwen PLT source-control controls disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  scripts/research/run_cross_model_mask_shuffled_negative_control_smoke.py \\
  scripts/research/run_cross_model_wrong_target_negative_control_smoke.py

echo '--- Qwen2.5-VL PLT source-control shifted+shuffled controls ({suffix}) ---'
.venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/qwen2p5vl_plt_source_control_controls_work_{suffix}" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$RUN_MANIFEST" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask \\
  --position-group top_hidden_delta_plus_answer_adjacent \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage3_qwen2p5vl_plt_source_control_{suffix}.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_plt_source_control_{suffix}.csv"
"""


def _fetch_outputs(sftp, runner, pack: str) -> None:
    suffix = f"{pack}_controls_full"
    for name in [
        f"stage3_qwen2p5vl_plt_source_control_{suffix}.json",
        f"stage3_qwen2p5vl_plt_source_control_{suffix}.csv",
    ]:
        try:
            sftp.get(f"{runner.REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL-PLT source-control shifted/shuffled controls only.")
    parser.add_argument("--pack", choices=["primary", "strict"], required=True)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    args = parser.parse_args()

    runner = _load_paperpack_runner()
    base = runner._load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    for script_name in [
        "run_stage2o_cross_model_source_control_probe.py",
        "run_stage2o_attribution_weighted_feature_bridge.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_cross_model_mask_shuffled_negative_control_smoke.py",
        "run_cross_model_wrong_target_negative_control_smoke.py",
    ]:
        runner._upload_research_script(base, sftp, script_name)
    runner._upload_pack_assets(base, sftp, args.pack, "full")

    remote_script = f"{runner.REMOTE_STAGE}/run_stage3_qwen_plt_source_control_controls_{args.pack}.sh"
    base._mkdir_p(sftp, runner.REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(runner, args.pack).replace("\r\n", "\n"))
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
        if time.time() - start > args.timeout_seconds:
            stdout.channel.close()
            raise TimeoutError("remote Qwen PLT source-control controls command exceeded timeout")

    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}", flush=True)
    sftp = client.open_sftp()
    _fetch_outputs(sftp, runner, args.pack)
    sftp.close()
    client.close()
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
