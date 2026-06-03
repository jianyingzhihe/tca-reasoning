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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage6_gemma_hidden_to_plt_decomp"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
RUNS = {
    "primary": PAPERPACK / "paperpack72_primary_prompt_runs.csv",
    "strict": PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv",
}
PREFIX = "stage6_gemma_hidden_to_plt_decomp"


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
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
    rows = _read_csv(runs_path)
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
            key = (stem, name)
            if key in seen_masks:
                continue
            seen_masks.add(key)
            local_mask = Path(row.get("mask_dir", "")) / name
            if not local_mask.exists():
                continue
            if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{name}"):
                uploaded += 1
            else:
                skipped += 1
    print(f"uploaded {remote_name}: rows={len(rows)} uploaded={uploaded} skipped_existing={skipped}", flush=True)


def _stem(pack: str, mode: str, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"{PREFIX}_{pack}_{mode}{suffix}"


def _cache_evict_block() -> str:
    return r"""
echo '--- Stage6-016 cgroup memory before cache evict ---'
cat /sys/fs/cgroup/memory.max 2>/dev/null || true
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.stat 2>/dev/null | egrep '^(anon|file|active_file|inactive_file|slab)' || true

.venv/bin/python - <<'PY'
import os
import time

roots = [
    "/root/autodl-tmp/tca-reasoning/data/hf_cache",
    "/root/autodl-tmp/tca-reasoning/stage6_gemma_hidden_to_plt_decomp",
]
has_fadvise = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
count = bytes_seen = ok = err = 0
for root in roots:
    if not os.path.exists(root):
        continue
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                size = os.stat(path).st_size
            except OSError:
                continue
            if size < 1024 * 1024:
                continue
            count += 1
            bytes_seen += size
            if not has_fadvise:
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    ok += 1
                finally:
                    os.close(fd)
            except OSError:
                err += 1
print(f"[cache-evict] posix_fadvise={has_fadvise} files={count} bytes_seen={bytes_seen} ok={ok} err={err}", flush=True)
time.sleep(1)
PY

echo '--- Stage6-016 cgroup memory after cache evict ---'
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
cat /sys/fs/cgroup/memory.stat 2>/dev/null | egrep '^(anon|file|active_file|inactive_file|slab)' || true
"""


def _remote_script(
    *,
    pack: str,
    mode: str,
    tag: str,
    layers: str,
    position_groups: str,
    topks: str,
    resume: bool,
    checkpoint_every: int,
) -> str:
    stem = _stem(pack, mode, tag)
    runs_name = f"paperpack72_{pack}_prompt_runs.csv"
    max_prompt_runs = 6 if mode == "smoke" else 0
    resume_flag = "  --resume \\\n" if resume else ""
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
mkdir -p "$STAGE"

echo '--- Stage6-016 Gemma hidden-to-PLT decomp preflight ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

{_cache_evict_block()}

.venv/bin/python -m py_compile scripts/research/run_stage6_gemma_hidden_to_plt_decomposition.py

if [ -z "$GEMMA_MODEL" ]; then
  echo 'GEMMA_MODEL_CACHE_MISSING: hidden-to-PLT decomp skipped'
  exit 3
fi

echo '--- Stage6-016 Gemma hidden-to-PLT decomp {pack} ---'
.venv/bin/python -u scripts/research/run_stage6_gemma_hidden_to_plt_decomposition.py \\
  --model-name "$GEMMA_MODEL" \\
  --transcoder-ref tianhux2/gemma3-4b-it-plt \\
  --prompt-runs "$STAGE/{runs_name}" \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{stem}_raw.csv" \\
  --summary-json "$STAGE/{stem}_run.json" \\
  --layers {layers} \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --position-groups {position_groups} \\
  --top-ks {topks} \\
{resume_flag}  --checkpoint-every {checkpoint_every} \\
  --max-prompt-runs {max_prompt_runs}

echo '--- Stage6-016 Gemma hidden-to-PLT decomp done ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true
"""


def _fetch(sftp, pack: str, mode: str, tag: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    stem = _stem(pack, mode, tag)
    for suffix in ["raw.csv", "run.json"]:
        name = f"{stem}_{suffix}"
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def _status_command(pack: str, mode: str, tag: str) -> str:
    stem = _stem(pack, mode, tag)
    return f"""
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'stage6_gemma_hidden_to_plt_decomp|run_stage6_gemma_hidden_to_plt_decomposition|gemma3-4b-it-plt' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{stem}* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*hidden_to_plt* 2>/dev/null || true
tail -n 120 {REMOTE_STAGE}/logs/*hidden_to_plt* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo DISK
df -h /root/autodl-tmp
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage6-016 Gemma hidden-to-PLT decomposition on AutoDL.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="decomp_v1")
    parser.add_argument("--layers", default="1")
    parser.add_argument("--position-groups", default="visual+answer,top_hidden_delta_16,top_hidden_delta_32")
    parser.add_argument("--topks", default="8,16,32")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
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
        _stdin, stdout, stderr = client.exec_command(_status_command(args.pack, args.mode, args.tag))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch(sftp, args.pack, args.mode, args.tag)
        sftp.close()
        client.close()
        if not args.skip_analyze:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "local" / "analyze_stage6_hidden_to_plt_decomposition.py"),
                    "--mode",
                    args.mode,
                    "--tag",
                    args.tag,
                ],
                check=False,
            )
        return 0

    _upload_research_script(base, sftp, "run_stage6_gemma_hidden_to_plt_decomposition.py")
    _upload_prompt_runs(base, sftp, RUNS[args.pack], f"paperpack72_{args.pack}_prompt_runs.csv")
    remote_script = f"{REMOTE_STAGE}/run_{_stem(args.pack, args.mode, args.tag)}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                pack=args.pack,
                mode=args.mode,
                tag=args.tag,
                layers=args.layers,
                position_groups=args.position_groups,
                topks=args.topks,
                resume=args.resume,
                checkpoint_every=args.checkpoint_every,
            ).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        remote_log_dir = f"{REMOTE_STAGE}/logs"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log = f"{remote_log_dir}/hidden_to_plt_{args.pack}_{args.mode}_{args.tag}_{stamp}.log"
        cmd = f"mkdir -p {remote_log_dir}; nohup bash {remote_script} > {remote_log} 2>&1 < /dev/null & echo $!"
        _stdin, stdout, stderr = client.exec_command(cmd)
        pid = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        print({"detached_remote_pid": pid, "remote_log": remote_log, "remote_script": remote_script})
        if err:
            print(err, file=sys.stderr)
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
        if time.time() - start > args.timeout_seconds:
            timed_out = True
            stdout.channel.close()
            break
        time.sleep(0.5)
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = 124 if timed_out else stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")

    sftp = client.open_sftp()
    _fetch(sftp, args.pack, args.mode, args.tag)
    sftp.close()
    client.close()
    if exit_status == 0 and not args.skip_analyze:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage6_hidden_to_plt_decomposition.py"),
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
