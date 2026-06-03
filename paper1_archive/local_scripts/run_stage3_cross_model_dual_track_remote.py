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


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _upload_stage3_assets(base, sftp) -> None:
    for name in ["stage3_aligned24_manifest.csv", "stage3_aligned48_prompt_runs.csv", "stage3_asset_table.csv"]:
        base._put_file(sftp, LOCAL_STAGE / name, f"{REMOTE_STAGE}/{name}")
    # Upload only manifest-referenced images/masks. This keeps remote jobs self-contained.
    import csv

    with (LOCAL_STAGE / "stage3_aligned24_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image = Path(row["local_image_path"])
        mask_dir = Path(row["mask_dir"])
        image_name = Path(row["image_filename"]).name
        stem = Path(image_name).stem
        if image.exists():
            base._put_file(sftp, image, f"{REMOTE_ASSETS}/images/{image_name}")
        for mask_name in ["answer.png", "relate.png"]:
            local_mask = mask_dir / mask_name
            if local_mask.exists():
                base._put_file(sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}")
    print(f"uploaded Stage3 assets for {len(rows)} samples", flush=True)


def _remote_script(mode: str) -> str:
    smoke_limit = "--max-runs 6" if mode == "smoke" else ""
    source_control_block = "" if mode == "smoke" else f"""
echo '--- Stage3 Qwen2.5-VL PLT source-control full probe ---'
.venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/qwen2p5vl_plt_source_control_work" \\
  --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
  --run-manifest "$STAGE/stage3_aligned48_prompt_runs.csv" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask \\
  --position-group top_hidden_delta_plus_answer_adjacent \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage3_qwen2p5vl_plt_source_control.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_plt_source_control.csv"

echo '--- Stage3 Qwen2.5-VL CLT source-control full probe ---'
.venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/qwen2p5vl_clt_source_control_work" \\
  --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
  --run-manifest "$STAGE/stage3_aligned48_prompt_runs.csv" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask \\
  --position-group top_hidden_delta_plus_answer_adjacent \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage3_qwen2p5vl_clt_source_control.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_clt_source_control.csv"

echo '--- Stage3 optional LLaVA CLT source-control full probe ---'
if [ -d "$LLAVA_MODEL_DIR" ]; then
  .venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
    --model-family llava \\
    --model-name "$LLAVA_MODEL_DIR" \\
    --transcoder-ref KokosDev/llava15-7b-clt \\
    --annotation-roots "$ASSET_ROOT" \\
    --work-dir "$STAGE/llava15_clt_source_control_work" \\
    --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
    --run-manifest "$STAGE/stage3_aligned48_prompt_runs.csv" \\
    --layer 15 \\
    --mask-conditions answer_mask,union_mask \\
    --position-group top_hidden_delta_plus_answer_adjacent \\
    --top-k-features 8 \\
    --control-pool-size 2048 \\
    --out-json "$STAGE/stage3_llava15_clt_source_control.json" \\
    --out-csv "$STAGE/stage3_llava15_clt_source_control.csv"
else
  echo 'LLAVA_MODEL_DIR_MISSING: skipping LLaVA CLT source-control full probe'
fi
"""
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
LLAVA_MODEL_DIR=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf

echo '--- Stage3 disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  scripts/research/run_cross_model_wrong_target_negative_control_smoke.py \\
  scripts/research/run_stage3_asset_preflight.py \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py

echo '--- Stage3 asset preflight ---'
.venv/bin/python -u scripts/research/run_stage3_asset_preflight.py \\
  --out-json "$STAGE/stage3_asset_preflight.json" \\
  --out-csv "$STAGE/stage3_asset_preflight.csv"

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: Qwen feature bridge smoke skipped'
  exit 0
fi

echo '--- Stage3 Qwen2.5-VL PLT feature bridge smoke/audit ---'
.venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/qwen2p5vl_plt_feature_work" \\
  --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
  --run-manifest "$STAGE/stage3_aligned48_prompt_runs.csv" \\
  --layer 26 \\
  --mask-condition union_mask \\
  --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  {smoke_limit} \\
  --out-json "$STAGE/stage3_qwen2p5vl_plt_feature_union.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_plt_feature_union.csv"

echo '--- Stage3 Qwen2.5-VL CLT matched feature bridge smoke/audit ---'
.venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/qwen2p5vl_clt_feature_work" \\
  --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
  --run-manifest "$STAGE/stage3_aligned48_prompt_runs.csv" \\
  --layer 26 \\
  --mask-condition union_mask \\
  --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  {smoke_limit} \\
  --out-json "$STAGE/stage3_qwen2p5vl_clt_feature_union.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_clt_feature_union.csv"

echo '--- Stage3 optional LLaVA CLT auxiliary smoke ---'
if [ -d "$LLAVA_MODEL_DIR" ]; then
  .venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
    --model-family llava \\
    --model-name "$LLAVA_MODEL_DIR" \\
    --transcoder-ref KokosDev/llava15-7b-clt \\
    --annotation-roots "$ASSET_ROOT" \\
    --work-dir "$STAGE/llava15_clt_feature_work" \\
    --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
    --run-manifest "$STAGE/stage3_aligned48_prompt_runs.csv" \\
    --layer 15 \\
    --mask-condition union_mask \\
    --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
    --top-k-features 8 \\
    --control-pool-size 2048 \\
    {smoke_limit} \\
    --out-json "$STAGE/stage3_llava15_clt_feature_union.json" \\
    --out-csv "$STAGE/stage3_llava15_clt_feature_union.csv"
else
  echo 'LLAVA_MODEL_DIR_MISSING: skipping LLaVA CLT auxiliary smoke'
fi
{source_control_block}
"""


def _fetch_outputs(sftp) -> None:
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    names = [
        "stage3_asset_preflight.json",
        "stage3_asset_preflight.csv",
        "stage3_qwen2p5vl_plt_feature_union.json",
        "stage3_qwen2p5vl_plt_feature_union.csv",
        "stage3_qwen2p5vl_clt_feature_union.json",
        "stage3_qwen2p5vl_clt_feature_union.csv",
        "stage3_llava15_clt_feature_union.json",
        "stage3_llava15_clt_feature_union.csv",
        "stage3_qwen2p5vl_plt_source_control.json",
        "stage3_qwen2p5vl_plt_source_control.csv",
        "stage3_qwen2p5vl_clt_source_control.json",
        "stage3_qwen2p5vl_clt_source_control.csv",
        "stage3_llava15_clt_source_control.json",
        "stage3_llava15_clt_source_control.csv",
    ]
    for name in names:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_STAGE / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage3 cross-model dual-track smoke/full job on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
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
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_cross_model_wrong_target_negative_control_smoke.py",
        "run_stage3_asset_preflight.py",
        "run_stage2o_attribution_weighted_feature_bridge.py",
        "run_stage2o_cross_model_source_control_probe.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    _upload_stage3_assets(base, sftp)
    remote_script = f"{REMOTE_STAGE}/run_stage3_cross_model_dual_track_{args.mode}.sh"
    base._mkdir_p(sftp, REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.mode).replace("\r\n", "\n"))
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
            raise TimeoutError("remote Stage3 command exceeded timeout")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")

    sftp = client.open_sftp()
    _fetch_outputs(sftp)
    sftp.close()
    client.close()
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
