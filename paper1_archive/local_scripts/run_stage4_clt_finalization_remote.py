#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import socket
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_clt_finalization"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"


ASSETS = {
    "qwen_clt": {
        "label": "qwen2p5vl_clt",
        "family": "qwen",
        "transcoder": "KokosDev/qwen2p5vl-7b-clt",
        "layers": [26],
        "model_expr": '"$QWEN_MODEL"',
        "require": '[ -n "$QWEN_MODEL" ]',
    },
    "llava_clt": {
        "label": "llava15_clt",
        "family": "llava",
        "transcoder": "KokosDev/llava15-7b-clt",
        "layers": [12, 15, 18, 21],
        "model_expr": '"$LLAVA_MODEL"',
        "require": '[ -n "$LLAVA_MODEL" ]',
    },
}


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


def _asset_list(asset: str) -> list[str]:
    return ["qwen_clt", "llava_clt"] if asset == "both" else [asset]


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _put_if_missing(base, sftp, local: Path, remote: str) -> bool:
    try:
        sftp.stat(remote)
        return False
    except OSError:
        base._put_file(sftp, local, remote)
        return True


def _upload_pack_assets(base, sftp, pack: str, mode: str) -> None:
    sample_manifest, prompt_manifest = _manifest_paths(pack)
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._put_file(sftp, sample_manifest, f"{REMOTE_STAGE}/paperpack_{pack}_manifest.csv")
    base._put_file(sftp, prompt_manifest, f"{REMOTE_STAGE}/paperpack_{pack}_prompt_runs.csv")
    with sample_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if mode == "smoke":
        with prompt_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            prompt_rows = list(csv.DictReader(handle))[:6]
        keep_ids = {row["sample_id"] for row in prompt_rows}
        rows = [row for row in rows if row["sample_id"] in keep_ids]
    uploaded = 0
    skipped = 0
    for row in rows:
        image = Path(row["local_image_path"])
        image_name = Path(row["image_filename"]).name
        stem = Path(image_name).stem
        if _put_if_missing(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
            uploaded += 1
        else:
            skipped += 1
        mask_dir = Path(row["mask_dir"])
        for mask_name in ["answer.png", "relate.png", "union.png", "shifted.png", "shuffled.png"]:
            local_mask = mask_dir / mask_name
            if local_mask.exists():
                if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}"):
                    uploaded += 1
                else:
                    skipped += 1
    print(
        f"uploaded Stage4 CLT {pack} assets: samples={len(rows)} files_uploaded={uploaded} files_skipped={skipped}",
        flush=True,
    )


def _asset_block(asset_key: str, pack: str, mode: str, topks: list[int]) -> str:
    cfg = ASSETS[asset_key]
    label = cfg["label"]
    family = cfg["family"]
    transcoder = cfg["transcoder"]
    model_expr = cfg["model_expr"]
    require = cfg["require"]
    max_runs = "--max-runs 6" if mode == "smoke" else ""
    chunks = [
        f"""
echo '--- Stage4 CLT finalization availability: {label} ---'
if ! {require}; then
  echo 'STAGE4_CLT_ASSET_MISSING: {label}'
else
"""
    ]
    for layer in cfg["layers"]:
        for topk in topks:
            suffix = f"{label}_{pack}_{mode}_L{layer}_topK{topk}"
            chunks.append(
                f"""
  echo '--- {label} feature bridge {suffix} ---'
  .venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
    --model-family {family} \\
    --model-name {model_expr} \\
    --transcoder-ref {transcoder} \\
    --annotation-roots "$ASSET_ROOT" \\
    --work-dir "$STAGE/{suffix}_feature_work" \\
    --sample-manifest "$SAMPLE_MANIFEST" \\
    --run-manifest "$RUN_MANIFEST" \\
    --layer {layer} \\
    --mask-condition union_mask \\
    --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
    --top-k-features {topk} \\
    --control-pool-size 2048 \\
    {max_runs} \\
    --out-json "$STAGE/stage4_{suffix}_feature_union.json" \\
    --out-csv "$STAGE/stage4_{suffix}_feature_union.csv" || true

  echo '--- {label} source-control {suffix} ---'
  .venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
    --model-family {family} \\
    --model-name {model_expr} \\
    --transcoder-ref {transcoder} \\
    --annotation-roots "$ASSET_ROOT" \\
    --work-dir "$STAGE/{suffix}_source_control_work" \\
    --sample-manifest "$SAMPLE_MANIFEST" \\
    --run-manifest "$RUN_MANIFEST" \\
    --layer {layer} \\
    --mask-conditions answer_mask,union_mask \\
    --position-group top_hidden_delta_plus_answer_adjacent \\
    --top-k-features {topk} \\
    --control-pool-size 2048 \\
    {max_runs} \\
    --out-json "$STAGE/stage4_{suffix}_source_control.json" \\
    --out-csv "$STAGE/stage4_{suffix}_source_control.csv" || true
"""
            )
    chunks.append("fi\n")
    return "\n".join(chunks)


def _remote_script(asset: str, pack: str, mode: str, topks: list[int]) -> str:
    asset_blocks = "\n".join(_asset_block(key, pack, mode, topks) for key in _asset_list(asset))
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
SAMPLE_MANIFEST="$STAGE/paperpack_{pack}_manifest.csv"
RUN_MANIFEST="$STAGE/paperpack_{pack}_prompt_runs.csv"
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
LLAVA_LOCAL=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf
LLAVA_HF=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/* 2>/dev/null | head -n 1 || true)
if [ -d "$LLAVA_LOCAL" ]; then
  LLAVA_MODEL="$LLAVA_LOCAL"
elif [ -n "$LLAVA_HF" ]; then
  LLAVA_MODEL="$LLAVA_HF"
else
  LLAVA_MODEL="llava-hf/llava-1.5-7b-hf"
fi

echo '--- Stage4 CLT finalization disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  scripts/research/run_cross_model_mask_shuffled_negative_control_smoke.py \\
  scripts/research/run_cross_model_wrong_target_negative_control_smoke.py

{asset_blocks}
"""


def _fetch_outputs(sftp, asset: str, pack: str, mode: str, topks: list[int]) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    for key in _asset_list(asset):
        cfg = ASSETS[key]
        label = cfg["label"]
        for layer in cfg["layers"]:
            for topk in topks:
                suffix = f"{label}_{pack}_{mode}_L{layer}_topK{topk}"
                for tail in ["feature_union.json", "feature_union.csv", "source_control.json", "source_control.csv"]:
                    name = f"stage4_{suffix}_{tail}"
                    try:
                        sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                        print(f"fetched {name}", flush=True)
                    except FileNotFoundError:
                        print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4 CLT finalization jobs on AutoDL.")
    parser.add_argument("--asset", choices=["qwen_clt", "llava_clt", "both"], default="qwen_clt")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--topks", default="1,4,8,16,32")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args()
    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    for script_name in [
        "run_stage2o_attribution_weighted_feature_bridge.py",
        "run_stage2o_cross_model_source_control_probe.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_cross_model_mask_shuffled_negative_control_smoke.py",
        "run_cross_model_wrong_target_negative_control_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    _upload_pack_assets(base, sftp, args.pack, args.mode)
    remote_script = f"{REMOTE_STAGE}/run_stage4_clt_finalization_{args.asset}_{args.pack}_{args.mode}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.asset, args.pack, args.mode, topks).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
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
    _fetch_outputs(sftp, args.asset, args.pack, args.mode, topks)
    sftp.close()
    client.close()
    if timed_out:
        raise TimeoutError("remote Stage4 CLT finalization command exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())

