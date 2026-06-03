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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage5_clt_heterogeneity"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage5" / "cross_model"
PREFIX = "stage5_clt_heterogeneity"

ASSETS = {
    "qwen_clt": {
        "label": "qwen2p5vl_clt",
        "family": "qwen",
        "transcoder": "KokosDev/qwen2p5vl-7b-clt",
        "default_layers": [0, 7, 14, 21, 26],
        "all_layers": list(range(27)),
        "model_expr": '"$QWEN_MODEL"',
        "require": '[ -n "$QWEN_MODEL" ]',
    },
    "llava_clt": {
        "label": "llava15_clt",
        "family": "llava",
        "transcoder": "KokosDev/llava15-7b-clt",
        "default_layers": [0, 12, 15, 18, 21, 30],
        "all_layers": [0, 12, 15, 18, 21, 30],
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
        return LOCAL_PAPERPACK / "paperpack72_primary_manifest.csv", LOCAL_PAPERPACK / "paperpack72_primary_prompt_runs.csv"
    if pack == "strict":
        return (
            LOCAL_PAPERPACK / "paperpack72_strict_sensitivity_manifest.csv",
            LOCAL_PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv",
        )
    raise ValueError(f"unknown pack: {pack}")


def _parse_layers(asset: str, raw: str) -> list[int]:
    if raw == "all":
        return ASSETS[asset]["all_layers"]
    if raw == "default":
        return ASSETS[asset]["default_layers"]
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _put_if_missing(base, sftp, local: Path, remote: str) -> bool:
    try:
        sftp.stat(remote)
        return False
    except OSError:
        base._put_file(sftp, local, remote)
        return True


def _upload_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _upload_pack_assets(base, sftp, pack: str, mode: str) -> None:
    sample_manifest, prompt_manifest = _manifest_paths(pack)
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/exported_masks")
    base._put_file(sftp, sample_manifest, f"{REMOTE_STAGE}/paperpack_{pack}_manifest.csv")
    base._put_file(sftp, prompt_manifest, f"{REMOTE_STAGE}/paperpack_{pack}_prompt_runs.csv")
    rows = _read_csv(sample_manifest)
    if mode == "smoke":
        prompt_rows = _read_csv(prompt_manifest)[:6]
        keep_ids = {row["sample_id"] for row in prompt_rows}
        rows = [row for row in rows if row["sample_id"] in keep_ids]
    uploaded = 0
    skipped = 0
    for row in rows:
        image = Path(row["local_image_path"])
        image_name = Path(row["image_filename"]).name
        stem = Path(image_name).stem
        if image.exists() and _put_if_missing(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
            uploaded += 1
        else:
            skipped += 1
        mask_dir = Path(row["mask_dir"])
        for mask_name in ["answer.png", "relate.png", "union.png", "shifted.png", "shuffled.png"]:
            local_mask = mask_dir / mask_name
            if local_mask.exists() and _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}"):
                uploaded += 1
            else:
                skipped += 1
    print(f"uploaded Stage5 {pack} assets: samples={len(rows)} uploaded={uploaded} skipped={skipped}", flush=True)


def _artifact_names(asset: str, pack: str, mode: str, layer: int, topk: int) -> list[str]:
    label = ASSETS[asset]["label"]
    stem = f"{PREFIX}_{label}_{pack}_{mode}_L{layer}_topK{topk}"
    return [
        f"{stem}_feature_union.csv",
        f"{stem}_feature_union.json",
        f"{stem}_source_control.csv",
        f"{stem}_source_control.json",
    ]


def _remote_asset_block(asset: str, pack: str, mode: str, layers: list[int], topks: list[int], resume: bool) -> str:
    cfg = ASSETS[asset]
    max_runs = "--max-runs 6" if mode == "smoke" else ""
    chunks = [
        f"""
echo '--- Stage5 CLT availability: {cfg["label"]} ---'
if ! {cfg["require"]}; then
  echo 'STAGE5_CLT_ASSET_MISSING: {cfg["label"]}'
else
"""
    ]
    for layer in layers:
        for topk in topks:
            label = cfg["label"]
            stem = f"{PREFIX}_{label}_{pack}_{mode}_L{layer}_topK{topk}"
            skip_feature = f'[ -s "$STAGE/{stem}_feature_union.csv" ]' if resume else "false"
            skip_source = f'[ -s "$STAGE/{stem}_source_control.csv" ]' if resume else "false"
            chunks.append(
                f"""
  echo '--- {label} feature bridge {stem} ---'
  if {skip_feature}; then
    echo 'resume_skip_feature {stem}'
  else
    .venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
      --model-family {cfg["family"]} \\
      --model-name {cfg["model_expr"]} \\
      --transcoder-ref {cfg["transcoder"]} \\
      --annotation-roots "$ASSET_ROOT" \\
      --work-dir "$STAGE/{stem}_feature_work" \\
      --sample-manifest "$SAMPLE_MANIFEST" \\
      --run-manifest "$RUN_MANIFEST" \\
      --layer {layer} \\
      --mask-condition union_mask \\
      --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
      --top-k-features {topk} \\
      --control-pool-size 2048 \\
      {max_runs} \\
      --out-json "$STAGE/{stem}_feature_union.json" \\
      --out-csv "$STAGE/{stem}_feature_union.csv" || true
  fi

  echo '--- {label} source-control {stem} ---'
  if {skip_source}; then
    echo 'resume_skip_source {stem}'
  else
    .venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
      --model-family {cfg["family"]} \\
      --model-name {cfg["model_expr"]} \\
      --transcoder-ref {cfg["transcoder"]} \\
      --annotation-roots "$ASSET_ROOT" \\
      --work-dir "$STAGE/{stem}_source_control_work" \\
      --sample-manifest "$SAMPLE_MANIFEST" \\
      --run-manifest "$RUN_MANIFEST" \\
      --layer {layer} \\
      --mask-conditions answer_mask,union_mask \\
      --position-group top_hidden_delta_plus_answer_adjacent \\
      --top-k-features {topk} \\
      --control-pool-size 2048 \\
      {max_runs} \\
      --out-json "$STAGE/{stem}_source_control.json" \\
      --out-csv "$STAGE/{stem}_source_control.csv" || true
  fi
"""
            )
    chunks.append("fi\n")
    return "\n".join(chunks)


def _remote_script(asset: str, pack: str, mode: str, layers: list[int], topks: list[int], resume: bool) -> str:
    block = _remote_asset_block(asset, pack, mode, layers, topks, resume)
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
if [ -d "$LLAVA_LOCAL" ]; then LLAVA_MODEL="$LLAVA_LOCAL"; elif [ -n "$LLAVA_HF" ]; then LLAVA_MODEL="$LLAVA_HF"; else LLAVA_MODEL="llava-hf/llava-1.5-7b-hf"; fi
mkdir -p "$STAGE"
echo '--- Stage5 CLT disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
.venv/bin/python -m py_compile \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py
{block}
"""


def _status_command(asset: str, pack: str, mode: str) -> str:
    return f"""
STAGE={REMOTE_STAGE}
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args --cols 260 | grep -E 'stage5_clt|stage2o_attribution|stage2o_cross_model' | grep -v grep || true
echo FILES
ls -lh "$STAGE"/{PREFIX}_*_{pack}_{mode}_L* 2>/dev/null | tail -120 || true
echo LOGS
ls -lh "$STAGE"/logs/* 2>/dev/null | tail -40 || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""


def _fetch(sftp, asset: str, pack: str, mode: str, layers: list[int], topks: list[int]) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    fetched = 0
    missing = 0
    for layer in layers:
        for topk in topks:
            for name in _artifact_names(asset, pack, mode, layer, topk):
                try:
                    sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                    fetched += 1
                except FileNotFoundError:
                    missing += 1
    print(f"fetch complete: fetched={fetched} missing={missing}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage5 CLT heterogeneity layer screen.")
    parser.add_argument("--asset", choices=["qwen_clt", "llava_clt"], required=True)
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "screen", "full", "strict-confirm"], default="smoke")
    parser.add_argument("--layers", default="default")
    parser.add_argument("--topks", default="1,8,32")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    topks = [int(part.strip()) for part in args.topks.split(",") if part.strip()]
    layers = _parse_layers(args.asset, args.layers)

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.asset, args.pack, args.mode))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err)
        client.close()
        return 0
    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch(sftp, args.asset, args.pack, args.mode, layers, topks)
        sftp.close()
        client.close()
        return 0
    for script in ["run_stage2o_attribution_weighted_feature_bridge.py", "run_stage2o_cross_model_source_control_probe.py"]:
        _upload_script(base, sftp, script)
    _upload_pack_assets(base, sftp, args.pack, args.mode)
    remote_script = f"{REMOTE_STAGE}/run_stage5_clt_{args.asset}_{args.pack}_{args.mode}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.asset, args.pack, args.mode, layers, topks, args.resume).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        log_dir = f"{REMOTE_STAGE}/logs"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = f"{log_dir}/stage5_clt_{args.asset}_{args.pack}_{args.mode}_{stamp}.log"
        cmd = f"mkdir -p {log_dir}; nohup bash {remote_script} > {log} 2>&1 < /dev/null & echo $!"
        _stdin, stdout, stderr = client.exec_command(cmd)
        print({"detached_remote_pid": stdout.read().decode().strip(), "remote_log": log, "remote_script": remote_script})
        err = stderr.read().decode("utf-8", errors="replace").strip()
        if err:
            print(err)
        client.close()
        return 0

    stdout = client.exec_command(f"bash {remote_script}", get_pty=True)[1]
    stdout.channel.settimeout(0.0)
    start = time.time()
    timed_out = False
    while not stdout.channel.exit_status_ready():
        try:
            data = stdout.channel.recv(8192)
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
    exit_status = 124 if timed_out else stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}", flush=True)
    sftp = client.open_sftp()
    _fetch(sftp, args.asset, args.pack, args.mode, layers, topks)
    sftp.close()
    client.close()
    if timed_out:
        raise TimeoutError("Stage5 CLT command timed out")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
