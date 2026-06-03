#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage6_node_generation_bridge"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
STAGE3_CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"
PREFIX = "stage6_node_generation_bridge"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner",
        ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _stem(mode: str, tag: str) -> str:
    return f"{PREFIX}_{mode}_{tag}"


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


def _put_if_exists(base, sftp, local: Path, remote: str) -> bool:
    if not local.exists():
        return False
    base._put_file(sftp, local, remote)
    return True


def _upload_gemma_assets(base, sftp, manifest: Path) -> None:
    rows = _read_csv(manifest)
    uploaded = 0
    for row in rows:
        image = Path(row.get("local_image_path", ""))
        image_name = Path(row.get("image_filename", "")).name
        if image_name and _put_if_exists(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
            uploaded += 1
        stem = Path(image_name).stem
        mask_dir = Path(row.get("mask_dir", ""))
        for mask_name in ["answer.png", "union.png", "shifted.png", "shuffled.png"]:
            if _put_if_exists(base, sftp, mask_dir / mask_name, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}"):
                uploaded += 1
    print(f"uploaded Gemma Stage6-022 assets: rows={len(rows)} files={uploaded}", flush=True)


def _upload_qwen_assets(base, sftp, qwen_manifest: Path) -> None:
    rows = _read_csv(qwen_manifest)
    sample_ids = {row.get("sample_id", "") for row in rows}
    sample_manifest = STAGE3_CROSS / "stage3_aligned24_manifest.csv"
    base._put_file(sftp, sample_manifest, f"{REMOTE_STAGE}/stage3_aligned24_manifest.csv")
    samples = [row for row in _read_csv(sample_manifest) if row.get("sample_id", "") in sample_ids]
    uploaded = 0
    for row in samples:
        image = Path(row.get("local_image_path", ""))
        image_name = Path(row.get("image_filename", "")).name
        if image_name and _put_if_exists(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
            uploaded += 1
        stem = Path(image_name).stem
        mask_dir = Path(row.get("mask_dir", ""))
        for mask_name in ["answer.png", "relate.png", "shifted.png", "shuffled.png"]:
            if _put_if_exists(base, sftp, mask_dir / mask_name, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}"):
                uploaded += 1
    print(f"uploaded Qwen Stage6-022 assets: samples={len(samples)} files={uploaded}", flush=True)


def _remote_script(mode: str, tag: str, max_new_tokens: int) -> str:
    stem = _stem(mode, tag)
    max_cases = 1 if mode == "smoke" else 0
    max_pairs = 1 if mode == "smoke" else 0
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
GEMMA_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--google--gemma-3-4b-it/snapshots/* 2>/dev/null | head -n 1 || true)
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
mkdir -p "$STAGE"

echo '--- Stage6-022 node-generation bridge preflight ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage6_gemma_hidden_generation_bridge.py \\
  scripts/research/run_stage6_qwen_hidden_generation_bridge.py \\
  scripts/research/run_stage6_gemma_hidden_to_plt_decomposition.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage3_qwen_multifeature_sequence_bridge.py

if [ -n "$GEMMA_MODEL" ]; then
  echo '--- Stage6-022 Gemma hidden generation bridge ---'
  .venv/bin/python -u scripts/research/run_stage6_gemma_hidden_generation_bridge.py \\
    --model-name "$GEMMA_MODEL" \\
    --manifest "$STAGE/{stem}_gemma_manifest.csv" \\
    --image-root "$ASSET_ROOT/images" \\
    --mask-root "$ASSET_ROOT/exported_masks" \\
    --layers 1 \\
    --bridge-operators hidden_residual,plt_topk_reconstruction,plt_reconstruction_error \\
    --transcoder-ref tianhux2/gemma3-4b-it-plt \\
    --topks 8,16,32 \\
    --max-cases {max_cases} \\
    --max-new-tokens {max_new_tokens} \\
    --out-json "$STAGE/{stem}_gemma.json" \\
    --out-csv "$STAGE/{stem}_gemma.csv"
else
  echo 'GEMMA_MODEL_CACHE_MISSING: skipping Gemma bridge'
fi

if [ -n "$QWEN_MODEL" ] && grep -q 'qwen2p5vl_plt' "$STAGE/{stem}_qwen_manifest.csv"; then
  echo '--- Stage6-022 Qwen PLT generation bridge ---'
  .venv/bin/python -u scripts/research/run_stage3_qwen_multifeature_sequence_bridge.py \\
    --model-name "$QWEN_MODEL" \\
    --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
    --asset-id qwen2p5vl_plt \\
    --annotation-roots "$ASSET_ROOT" \\
    --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
    --bridge-manifest "$STAGE/{stem}_qwen_manifest.csv" \\
    --topks 1,4,8,16,32 \\
    --max-pairs {max_pairs} \\
    --max-new-tokens {max_new_tokens} \\
    --out-json "$STAGE/{stem}_qwen2p5vl_plt.json" \\
    --out-csv "$STAGE/{stem}_qwen2p5vl_plt.csv"
fi

if [ -n "$QWEN_MODEL" ] && grep -q 'qwen2p5vl_clt' "$STAGE/{stem}_qwen_manifest.csv"; then
  echo '--- Stage6-022 Qwen CLT generation bridge ---'
  .venv/bin/python -u scripts/research/run_stage3_qwen_multifeature_sequence_bridge.py \\
    --model-name "$QWEN_MODEL" \\
    --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
    --asset-id qwen2p5vl_clt \\
    --annotation-roots "$ASSET_ROOT" \\
    --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
    --bridge-manifest "$STAGE/{stem}_qwen_manifest.csv" \\
    --topks 1,4,8,16,32 \\
    --max-pairs {max_pairs} \\
    --max-new-tokens {max_new_tokens} \\
    --out-json "$STAGE/{stem}_qwen2p5vl_clt.json" \\
    --out-csv "$STAGE/{stem}_qwen2p5vl_clt.csv"
fi

if [ -n "$QWEN_MODEL" ]; then
  echo '--- Stage6-022 Qwen hidden generation bridge ---'
  .venv/bin/python -u scripts/research/run_stage6_qwen_hidden_generation_bridge.py \\
    --model-name "$QWEN_MODEL" \\
    --annotation-roots "$ASSET_ROOT" \\
    --sample-manifest "$STAGE/stage3_aligned24_manifest.csv" \\
    --bridge-manifest "$STAGE/{stem}_qwen_manifest.csv" \\
    --layers 14 \\
    --position-group top_hidden_delta \\
    --top-delta-count 32 \\
    --max-pairs {max_pairs} \\
    --max-new-tokens {max_new_tokens} \\
    --out-json "$STAGE/{stem}_qwen2p5vl_hidden.json" \\
    --out-csv "$STAGE/{stem}_qwen2p5vl_hidden.csv"
fi

echo '--- Stage6-022 node-generation bridge done ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true
"""


def _status_command(mode: str, tag: str) -> str:
    stem = _stem(mode, tag)
    return f"""
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'stage6_node_generation_bridge|run_stage6_gemma_hidden_generation_bridge|run_stage6_qwen_hidden_generation_bridge|run_stage3_qwen_multifeature_sequence_bridge' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{stem}* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*node_generation* 2>/dev/null || true
tail -n 120 {REMOTE_STAGE}/logs/*node_generation* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo DISK
df -h /root/autodl-tmp
"""


def _fetch_outputs(sftp, mode: str, tag: str) -> None:
    stem = _stem(mode, tag)
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    for name in [
        f"{stem}_gemma.json",
        f"{stem}_gemma.csv",
        f"{stem}_qwen2p5vl_hidden.json",
        f"{stem}_qwen2p5vl_hidden.csv",
        f"{stem}_qwen2p5vl_plt.json",
        f"{stem}_qwen2p5vl_plt.csv",
        f"{stem}_qwen2p5vl_clt.json",
        f"{stem}_qwen2p5vl_clt.csv",
    ]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage6-022 node-to-generation bridge on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="defensive_v1")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)

    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.mode, args.tag))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch_outputs(sftp, args.mode, args.tag)
        sftp.close()
        client.close()
        if not args.skip_analyze:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "local" / "analyze_stage6_node_generation_bridge.py"),
                    "--mode",
                    args.mode,
                    "--tag",
                    args.tag,
                ],
                check=False,
            )
        return 0

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "local" / "build_stage6_node_generation_bridge_manifest.py"),
            "--mode",
            args.mode,
            "--tag",
            args.tag,
        ],
        check=True,
    )
    stem = _stem(args.mode, args.tag)
    gemma_manifest = LOCAL_CROSS / f"{stem}_gemma_manifest.csv"
    qwen_manifest = LOCAL_CROSS / f"{stem}_qwen_manifest.csv"

    for script_name in [
        "run_stage6_gemma_hidden_generation_bridge.py",
        "run_stage6_qwen_hidden_generation_bridge.py",
        "run_stage6_gemma_hidden_to_plt_decomposition.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_stage2o_attribution_weighted_feature_bridge.py",
        "run_stage3_qwen_multifeature_sequence_bridge.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._put_file(sftp, gemma_manifest, f"{REMOTE_STAGE}/{gemma_manifest.name}")
    base._put_file(sftp, qwen_manifest, f"{REMOTE_STAGE}/{qwen_manifest.name}")
    _upload_gemma_assets(base, sftp, gemma_manifest)
    _upload_qwen_assets(base, sftp, qwen_manifest)

    remote_script = f"{REMOTE_STAGE}/run_{stem}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.mode, args.tag, args.max_new_tokens).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        log = f"{REMOTE_STAGE}/logs/{stem}_node_generation.log"
        command = f"mkdir -p {REMOTE_STAGE}/logs && nohup bash {remote_script} > {log} 2>&1 & echo $!"
        _stdin, stdout, stderr = client.exec_command(command)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

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
            raise TimeoutError("remote Stage6-022 node-generation bridge exceeded timeout")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")
    sftp = client.open_sftp()
    _fetch_outputs(sftp, args.mode, args.tag)
    sftp.close()
    client.close()
    if not args.skip_analyze:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage6_node_generation_bridge.py"),
                "--mode",
                args.mode,
                "--tag",
                args.tag,
            ],
            check=False,
        )
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
