#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import posixpath
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage3_paperpack_plt"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _manifest_paths(pack: str) -> tuple[Path, Path]:
    if pack == "primary":
        return (
            LOCAL_PAPERPACK / "paperpack72_primary_manifest.csv",
            LOCAL_PAPERPACK / "paperpack72_primary_prompt_runs.csv",
        )
    if pack == "strict":
        return (
            LOCAL_PAPERPACK / "paperpack72_strict_sensitivity_manifest.csv",
            LOCAL_PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv",
        )
    raise ValueError(f"unsupported pack: {pack}")


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _upload_pack_assets(base, sftp, pack: str, mode: str) -> None:
    sample_manifest, prompt_manifest = _manifest_paths(pack)
    base._put_file(sftp, sample_manifest, f"{REMOTE_STAGE}/paperpack_{pack}_manifest.csv")
    base._put_file(sftp, prompt_manifest, f"{REMOTE_STAGE}/paperpack_{pack}_prompt_runs.csv")

    with sample_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if mode == "smoke":
        with prompt_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            prompt_rows = list(csv.DictReader(handle))[:6]
        smoke_ids = {row["sample_id"] for row in prompt_rows}
        rows_to_upload = [row for row in rows if row["sample_id"] in smoke_ids]
    else:
        rows_to_upload = rows

    for row in rows_to_upload:
        image = Path(row["local_image_path"])
        image_name = Path(row["image_filename"]).name
        stem = Path(image_name).stem
        base._put_file(sftp, image, f"{REMOTE_ASSETS}/images/{image_name}")
        mask_dir = Path(row["mask_dir"])
        for mask_name in ["answer.png", "relate.png", "union.png", "shifted.png", "shuffled.png"]:
            local_mask = mask_dir / mask_name
            if local_mask.exists():
                base._put_file(sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}")
    print(f"uploaded Stage3 paperpack {pack} assets for {len(rows_to_upload)}/{len(rows)} samples ({mode})", flush=True)


def _remote_script(pack: str, mode: str) -> str:
    max_runs = "--max-runs 6" if mode == "smoke" else ""
    suffix = f"{pack}_{mode}"
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
SAMPLE_MANIFEST="$STAGE/paperpack_{pack}_manifest.csv"
RUN_MANIFEST="$STAGE/paperpack_{pack}_prompt_runs.csv"

echo '--- Stage3 paperpack PLT disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage3_asset_preflight.py \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  scripts/research/run_cross_model_wrong_target_negative_control_smoke.py

echo '--- Stage3 PLT asset preflight ---'
.venv/bin/python -u scripts/research/run_stage3_asset_preflight.py \\
  --out-json "$STAGE/stage3_paperpack_asset_preflight_{suffix}.json" \\
  --out-csv "$STAGE/stage3_paperpack_asset_preflight_{suffix}.csv"

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: Qwen2.5-VL PLT paperpack run skipped'
  exit 3
fi

echo '--- Qwen2.5-VL PLT attribution-weighted feature bridge ({suffix}) ---'
.venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/qwen2p5vl_plt_feature_work_{suffix}" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$RUN_MANIFEST" \\
  --layer 26 \\
  --mask-condition union_mask \\
  --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  {max_runs} \\
  --out-json "$STAGE/stage3_qwen2p5vl_plt_feature_union_{suffix}.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_plt_feature_union_{suffix}.csv"

echo '--- Qwen2.5-VL PLT approximate source-control probe ({suffix}) ---'
.venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/qwen2p5vl_plt_source_control_work_{suffix}" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$RUN_MANIFEST" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask \\
  --position-group top_hidden_delta_plus_answer_adjacent \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  {max_runs} \\
  --out-json "$STAGE/stage3_qwen2p5vl_plt_source_control_{suffix}.json" \\
  --out-csv "$STAGE/stage3_qwen2p5vl_plt_source_control_{suffix}.csv"
"""


def _fetch_outputs(sftp, pack: str, mode: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    suffix = f"{pack}_{mode}"
    names = [
        f"stage3_paperpack_asset_preflight_{suffix}.json",
        f"stage3_paperpack_asset_preflight_{suffix}.csv",
        f"stage3_qwen2p5vl_plt_feature_union_{suffix}.json",
        f"stage3_qwen2p5vl_plt_feature_union_{suffix}.csv",
        f"stage3_qwen2p5vl_plt_source_control_{suffix}.json",
        f"stage3_qwen2p5vl_plt_source_control_{suffix}.csv",
    ]
    for name in names:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage3 paperpack PLT Qwen smoke/full job on AutoDL.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
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
        "run_stage3_asset_preflight.py",
        "run_stage2o_attribution_weighted_feature_bridge.py",
        "run_stage2o_cross_model_source_control_probe.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_cross_model_wrong_target_negative_control_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    _upload_pack_assets(base, sftp, args.pack, args.mode)

    remote_script = f"{REMOTE_STAGE}/run_stage3_paperpack_plt_{args.pack}_{args.mode}.sh"
    base._mkdir_p(sftp, REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.pack, args.mode).replace("\r\n", "\n"))
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
            raise TimeoutError("remote Stage3 paperpack PLT command exceeded timeout")

    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}", flush=True)
    sftp = client.open_sftp()
    _fetch_outputs(sftp, args.pack, args.mode)
    sftp.close()
    client.close()
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
