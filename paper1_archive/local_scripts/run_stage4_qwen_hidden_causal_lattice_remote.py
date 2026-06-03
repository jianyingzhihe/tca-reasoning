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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_decisive_route"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
PRIMARY_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
STRICT_RUNS = PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"


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


def _upload_prompt_runs(base, sftp, runs_path: Path, remote_name: str) -> None:
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/exported_masks")
    base._put_file(sftp, runs_path, f"{REMOTE_STAGE}/{remote_name}")
    rows = list(csv.DictReader(runs_path.open("r", encoding="utf-8-sig", newline="")))
    uploaded = 0
    skipped = 0
    seen_images: set[str] = set()
    seen_masks: set[tuple[str, str]] = set()
    for row in rows:
        image = Path(row.get("local_image_path", ""))
        image_name = Path(row.get("image_filename", "")).name
        if image.exists() and image_name and image_name not in seen_images:
            seen_images.add(image_name)
            if _put_if_missing(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
                uploaded += 1
            else:
                skipped += 1
        stem = Path(image_name).stem
        for name in ["answer.png", "union.png", "shifted.png", "shuffled.png"]:
            if (stem, name) in seen_masks:
                continue
            seen_masks.add((stem, name))
            local_mask = Path(row.get("mask_dir", "")) / name
            if not local_mask.exists():
                continue
            if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{name}"):
                uploaded += 1
            else:
                skipped += 1
    print(f"uploaded {remote_name}: rows={len(rows)} uploaded={uploaded} skipped_existing={skipped}", flush=True)


def _stem(pack: str, mode: str, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
    return f"stage4_qwen_decisive_route_hidden_{pack}_{mode}{suffix}"


def _pack_block(pack: str, mode: str, max_prompt_runs: int, layers: str, resume: bool, tag: str, checkpoint_every: int, max_clean_rank: int) -> str:
    runs_name = f"paperpack72_{pack}_prompt_runs.csv"
    stem = _stem(pack, mode, tag)
    resume_flag = "  --resume \\\n" if resume else ""
    return f"""
echo '--- Stage4-024 hidden causal lattice {pack} ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_hidden_causal_lattice.py \\
  --model-name "$QWEN_MODEL" \\
  --prompt-runs "$STAGE/{runs_name}" \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{stem}_raw.csv" \\
  --summary-json "$STAGE/{stem}_run.json" \\
  --layers {layers} \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --position-groups visual_span,answer_adjacent,top_hidden_delta,visual+answer \\
  --scales 1.0 \\
  --max-clean-rank {max_clean_rank} \\
{resume_flag}  --checkpoint-every {checkpoint_every} \\
  --max-prompt-runs {max_prompt_runs}
"""


def _remote_script(mode: str, packs: list[str], max_prompt_runs: int, layers: str, resume: bool, tag: str, checkpoint_every: int, max_clean_rank: int) -> str:
    blocks = "\n".join(_pack_block(pack, mode, max_prompt_runs, layers, resume, tag, checkpoint_every, max_clean_rank) for pack in packs)
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
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
mkdir -p "$STAGE"

echo '--- Stage4-024 Qwen decisive route disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_hidden_causal_lattice.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: hidden lattice skipped'
  exit 3
fi

{blocks}
"""


def _fetch(sftp, mode: str, packs: list[str], tag: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    for pack in packs:
        stem = _stem(pack, mode, tag)
        for suffix in ["raw.csv", "run.json"]:
            name = f"{stem}_{suffix}"
            try:
                sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                print(f"fetched {name}", flush=True)
            except FileNotFoundError:
                print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-024 Qwen hidden causal lattice on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--packs", default="primary")
    parser.add_argument("--layers", default="12,16,20,22,24,26,27")
    parser.add_argument("--max-prompt-runs", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tag", default="", help="Optional artifact suffix, e.g. alllayers.")
    parser.add_argument("--primary-runs", default="")
    parser.add_argument("--strict-runs", default="")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-clean-rank", type=int, default=10)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Launch the remote script under nohup so it survives local SSH/Codex disconnects.",
    )
    args = parser.parse_args()
    packs = [part.strip() for part in args.packs.split(",") if part.strip()]
    if args.mode == "full" and args.max_prompt_runs == 6:
        args.max_prompt_runs = 0
    for pack in packs:
        if pack not in {"primary", "strict"}:
            raise ValueError(f"unknown pack: {pack}")

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    for script in [
        "run_stage4_qwen_hidden_causal_lattice.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_stage4_qwen_causal_cutter_validation.py",
    ]:
        _upload_research_script(base, sftp, script)
    if "primary" in packs:
        primary_runs = Path(args.primary_runs) if args.primary_runs else PRIMARY_RUNS
        _upload_prompt_runs(base, sftp, primary_runs, "paperpack72_primary_prompt_runs.csv")
    if "strict" in packs:
        strict_runs = Path(args.strict_runs) if args.strict_runs else STRICT_RUNS
        _upload_prompt_runs(base, sftp, strict_runs, "paperpack72_strict_prompt_runs.csv")
    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_hidden_lattice_{args.mode}_{'_'.join(packs)}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                args.mode,
                packs,
                args.max_prompt_runs,
                args.layers,
                args.resume,
                args.tag,
                args.checkpoint_every,
                args.max_clean_rank,
            ).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log = f"{REMOTE_STAGE}/run_stage4_qwen_hidden_lattice_{args.mode}_{'_'.join(packs)}_{args.tag or 'untagged'}_{stamp}.log"
        detach_cmd = f"nohup bash {remote_script} > {remote_log} 2>&1 < /dev/null & echo $!"
        _stdin, stdout, stderr = client.exec_command(detach_cmd)
        pid = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        print(f"DETACHED_REMOTE_PID={pid}", flush=True)
        print(f"DETACHED_REMOTE_LOG={remote_log}", flush=True)
        if err:
            print(err, flush=True)
        client.close()
        return 0

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
    _fetch(sftp, args.mode, packs, args.tag)
    sftp.close()
    client.close()
    if not args.skip_analyze and exit_status == 0:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_decisive_route.py"),
                "--mode",
                args.mode,
                "--hidden-tag",
                args.tag,
            ],
            check=False,
        )
    if timed_out:
        raise TimeoutError("remote hidden lattice exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
