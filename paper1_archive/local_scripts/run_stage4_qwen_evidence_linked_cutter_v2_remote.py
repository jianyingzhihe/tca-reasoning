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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_evidence_linked_cutter_v2"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
LOCAL_MANIFEST = LOCAL_CROSS / "stage4_qwen_expanded_cutter_candidate_manifest.csv"


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
    base._put_file(sftp, manifest, f"{REMOTE_STAGE}/stage4_qwen_expanded_cutter_candidate_manifest.csv")
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8-sig", newline="")))
    uploaded = 0
    skipped = 0
    seen_images: set[str] = set()
    for row in rows:
        image = Path(row["local_image_path"])
        image_name = Path(row["image_filename"]).name
        if image.exists() and image_name not in seen_images:
            seen_images.add(image_name)
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
            if local_mask.exists():
                if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{out_name}"):
                    uploaded += 1
                else:
                    skipped += 1
    print(f"uploaded evidence-v2 assets: rows={len(rows)} uploaded={uploaded} skipped_existing={skipped}", flush=True)


def _prefix(mode: str, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"stage4_qwen_evidence_linked_cutter_v2_{mode}{suffix}"


def _remote_script(mode: str, tag: str, max_prompt_runs: int, selection: str) -> str:
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

echo '--- Stage4 Qwen evidence-link V2 disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_evidence_linked_cutter_v2.py \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: evidence-link V2 skipped'
  exit 3
fi

.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_linked_cutter_v2.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/stage4_qwen_expanded_cutter_candidate_manifest.csv" \\
  --selection {selection} \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/${{PREFIX}}_raw.csv" \\
  --summary-json "$STAGE/${{PREFIX}}_run.json" \\
  --layer 26 \\
  --top-ks 1,4,8,16 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-prompt-runs {max_prompt_runs}
"""


def _fetch(sftp, mode: str, tag: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(mode, tag)
    for name in [f"{prefix}_raw.csv", f"{prefix}_run.json"]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)
    if mode == "full" and not tag:
        raw = LOCAL_CROSS / f"{prefix}_raw.csv"
        if raw.exists():
            (LOCAL_CROSS / "stage4_qwen_evidence_linked_cutter_v2_raw.csv").write_bytes(raw.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-016 Qwen evidence-linked cutter V2 on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="")
    parser.add_argument("--selection", choices=["main", "pool", "all"], default="pool")
    parser.add_argument("--max-prompt-runs", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()
    if args.mode == "full" and args.max_prompt_runs == 6:
        args.max_prompt_runs = 0
    if not LOCAL_MANIFEST.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "local" / "build_stage4_qwen_expanded_cutter_manifest.py")], check=True)

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    for script in [
        "run_stage4_qwen_evidence_linked_cutter_v2.py",
        "run_stage4_qwen_causal_cutter_validation.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
    ]:
        _upload_research_script(base, sftp, script)
    _upload_assets(base, sftp, LOCAL_MANIFEST)
    script_suffix = f"_{args.tag}" if args.tag else ""
    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_evidence_linked_cutter_v2_{args.mode}{script_suffix}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.mode, args.tag, args.max_prompt_runs, args.selection).replace("\r\n", "\n"))
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
    _fetch(sftp, args.mode, args.tag)
    sftp.close()
    client.close()
    if not args.skip_analyze and args.mode == "full" and exit_status == 0:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_mainline_v2.py")], check=False)
    if timed_out:
        raise TimeoutError("remote Stage4 Qwen evidence-link V2 exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
