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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_source_tracing"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _remote_script(pack: str, mode: str, max_samples: int, top_features_per_sample: int) -> str:
    prefix = f"stage4_qwen_source_tracing_{pack}_{mode}"
    max_arg = f"--max-samples {max_samples}" if max_samples > 0 else "--max-samples 0"
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
PREFIX={prefix}
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
SAMPLE_MANIFEST="$STAGE/paperpack_{pack}_manifest.csv"
RUN_MANIFEST="$STAGE/paperpack_{pack}_prompt_runs.csv"

echo '--- Stage4 Qwen intervention repair disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_qwen_answer_aligned_intervention_smoke.py \\
  scripts/research/analyze_stage4_qwen_source_tracing.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: intervention repair skipped'
  exit 3
fi

.venv/bin/python -u scripts/research/run_qwen_answer_aligned_intervention_smoke.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$RUN_MANIFEST" \\
  --nodes-csv "$STAGE/${{PREFIX}}_nodes_detailed_controlled.csv" \\
  --meta-a "$STAGE/${{PREFIX}}_meta_a.csv" \\
  --meta-b "$STAGE/${{PREFIX}}_meta_b.csv" \\
  --image-root "$ASSET_ROOT/images" \\
  --out-csv "$STAGE/${{PREFIX}}_intervention.csv" \\
  --summary-json "$STAGE/${{PREFIX}}_intervention.json" \\
  --layer 26 \\
  --zeroing-modes subtract,add \\
  {max_arg} \\
  --top-features-per-sample {top_features_per_sample}

.venv/bin/python -u scripts/research/analyze_stage4_qwen_source_tracing.py \\
  --cross-dir "$STAGE" \\
  --pack {pack} \\
  --mode {mode}
"""


def _fetch_outputs(sftp, pack: str, mode: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    prefix = f"stage4_qwen_source_tracing_{pack}_{mode}"
    for tail in [
        "intervention.csv",
        "intervention.json",
        "analysis_summary.csv",
        "decision.json",
    ]:
        name = f"{prefix}_{tail}"
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair Stage4 Qwen full source-tracing intervention step.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--top-features-per-sample", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    args = parser.parse_args()

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    for script_name in [
        "run_qwen_answer_aligned_intervention_smoke.py",
        "analyze_stage4_qwen_source_tracing.py",
        "run_cross_model_feature_intervention_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_intervention_repair_{args.pack}_{args.mode}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(args.pack, args.mode, args.max_samples, args.top_features_per_sample).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    _stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
    stdout.channel.settimeout(0.0)
    stderr.channel.settimeout(0.0)
    start = time.time()
    timed_out = False
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
            timed_out = True
            stdout.channel.close()
            break
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = 124 if timed_out else stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}", flush=True)
    sftp = client.open_sftp()
    _fetch_outputs(sftp, args.pack, args.mode)
    sftp.close()
    client.close()
    if timed_out:
        raise TimeoutError("remote Stage4 Qwen intervention repair exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())

