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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_causal_cutter"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
LOCAL_MANIFEST = LOCAL_CROSS / "stage4_qwen_causal_cutter_candidate_manifest.csv"


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


def _put_if_missing(base, sftp, local: Path, remote: str) -> bool:
    try:
        sftp.stat(remote)
        return False
    except OSError:
        base._put_file(sftp, local, remote)
        return True


def _upload_assets(base, sftp, manifest: Path) -> None:
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/exported_masks")
    base._put_file(sftp, manifest, f"{REMOTE_STAGE}/stage4_qwen_causal_cutter_candidate_manifest.csv")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    uploaded = 0
    skipped = 0
    for row in rows:
        image = Path(row["local_image_path"])
        image_name = Path(row["image_filename"]).name
        if image.exists():
            if _put_if_missing(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
                uploaded += 1
            else:
                skipped += 1
        stem = Path(image_name).stem
        for key, out_name in [
            ("answer_mask_path", "answer.png"),
            ("union_mask_path", "union.png"),
            ("shifted_mask_path", "shifted.png"),
            ("shuffled_mask_path", "shuffled.png"),
        ]:
            local_mask = Path(row.get(key, ""))
            if not local_mask.exists():
                continue
            if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{out_name}"):
                uploaded += 1
            else:
                skipped += 1
    print(f"uploaded cutter assets: manifest_rows={len(rows)} uploaded={uploaded} skipped_existing={skipped}", flush=True)


def _prefix(mode: str, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
    return f"stage4_qwen_causal_cutter_validation_{mode}{suffix}"


def _remote_script(mode: str, tag: str, max_candidates: int, selection: str) -> str:
    prefix = _prefix(mode, tag)
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
mkdir -p "$STAGE"

echo '--- Stage4 Qwen causal cutter disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: cutter validation skipped'
  exit 3
fi

.venv/bin/python -u scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/stage4_qwen_causal_cutter_candidate_manifest.csv" \\
  --selection {selection} \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --work-dir "$STAGE/work_{mode}" \\
  --out-csv "$STAGE/${{PREFIX}}_raw.csv" \\
  --summary-json "$STAGE/${{PREFIX}}_run.json" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-candidates {max_candidates}
"""


def _fetch_outputs(sftp, mode: str, tag: str) -> list[Path]:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(mode, tag)
    fetched: list[Path] = []
    for name in [f"{prefix}_raw.csv", f"{prefix}_run.json"]:
        remote = f"{REMOTE_STAGE}/{name}"
        local = LOCAL_CROSS / name
        try:
            sftp.get(remote, str(local))
            fetched.append(local)
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)
    if mode == "full" and not tag:
        canonical = LOCAL_CROSS / "stage4_qwen_causal_cutter_validation_raw.csv"
        raw = LOCAL_CROSS / f"{prefix}_raw.csv"
        if raw.exists():
            canonical.write_bytes(raw.read_bytes())
            fetched.append(canonical)
            print(f"updated canonical {canonical.name}", flush=True)
    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-014 Qwen causal cutter validation on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="")
    parser.add_argument("--selection", choices=["main", "sensitivity", "all"], default="main")
    parser.add_argument("--candidate-manifest", type=Path, default=LOCAL_MANIFEST)
    parser.add_argument("--max-candidates", type=int, default=6, help="0 means all selected candidates.")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()
    if args.mode == "full" and args.max_candidates == 6:
        args.max_candidates = 0
    manifest = args.candidate_manifest
    if not manifest.exists() and manifest == LOCAL_MANIFEST:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "local" / "build_stage4_qwen_causal_cutter_manifest.py")],
            check=True,
        )
    if not manifest.exists():
        raise FileNotFoundError(f"candidate manifest not found: {manifest}")

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    for script_name in [
        "run_stage4_qwen_causal_cutter_validation.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    _upload_assets(base, sftp, manifest)

    script_suffix = f"_{args.tag}" if args.tag else ""
    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_causal_cutter_{args.mode}{script_suffix}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.mode, args.tag, args.max_candidates, args.selection).replace("\r\n", "\n"))
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
    _fetch_outputs(sftp, args.mode, args.tag)
    sftp.close()
    client.close()
    if not args.skip_analyze and args.mode == "full" and exit_status == 0:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_causal_cutter_validation.py")], check=False)
    if timed_out:
        raise TimeoutError("remote Stage4 Qwen causal cutter validation exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
