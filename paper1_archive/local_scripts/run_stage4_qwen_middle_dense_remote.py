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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_middle_dense"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
PRIMARY_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
STRICT_RUNS = PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"
PREFIX = "stage4_qwen_middle_dense"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_ints(raw: str) -> list[int]:
    if raw == "middle":
        return list(range(10, 18))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_groups(raw: str) -> list[str]:
    if raw == "all":
        return ["visual_only", "answer_adjacent_only", "visual_answer"]
    return [part.strip() for part in raw.split(",") if part.strip()]


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


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}", flush=True)


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
            if local_mask.exists():
                if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{name}"):
                    uploaded += 1
                else:
                    skipped += 1
    print(f"uploaded {remote_name}: rows={len(rows)} uploaded={uploaded} skipped={skipped}", flush=True)


def _stem(pack: str, mode: str, layer: int, group: str, artifact: str) -> str:
    return f"{PREFIX}_{pack}_{mode}_L{layer}_{group}_{artifact}"


def _block(
    pack: str,
    mode: str,
    layer: int,
    group: str,
    max_prompt_runs: int,
    candidate_pool_size: int,
    top_per_prompt_run: int,
    main_per_prompt_run: int,
    main_cap: int,
    max_candidates: int,
    scale: float,
) -> str:
    runs_name = f"paperpack72_{pack}_prompt_runs.csv"
    cand = _stem(pack, mode, layer, group, "candidates")
    zero = _stem(pack, mode, layer, group, "zeroing")
    grouped = _stem(pack, mode, layer, group, "group")
    return f"""
echo '--- Stage4-048 middle dense {pack} L{layer} {group} discovery ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_feature_discovery.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --prompt-runs "$STAGE/{runs_name}" \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{cand}.csv" \\
  --summary-json "$STAGE/{cand}.json" \\
  --layer {layer} \\
  --position-group {group} \\
  --candidate-pool-size {candidate_pool_size} \\
  --top-per-prompt-run {top_per_prompt_run} \\
  --main-per-prompt-run {main_per_prompt_run} \\
  --max-prompt-runs {max_prompt_runs}

echo '--- Stage4-048 cap main candidates to top {main_cap} ---'
.venv/bin/python - <<'PY'
import csv
from pathlib import Path
path = Path("$STAGE/{cand}.csv")
if path.exists() and path.stat().st_size:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    fields = list(rows[0].keys()) if rows else []
    main = [row for row in rows if row.get("include_main") == "1"]
    def score(row):
        try:
            return float(row.get("evidence_first_score") or 0)
        except ValueError:
            return 0.0
    keep = {{row.get("candidate_id", "") for row in sorted(main, key=score, reverse=True)[:{main_cap}]}}
    for row in rows:
        if row.get("include_main") == "1" and row.get("candidate_id", "") not in keep:
            row["include_main"] = "0"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"middle dense cap: rows={{len(rows)}} main_before={{len(main)}} main_after={{len(keep)}}")
PY

echo '--- Stage4-048 middle dense {pack} L{layer} {group} zeroing/restore scale {scale} ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{cand}.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --work-dir "$STAGE/work_{zero}" \\
  --out-csv "$STAGE/{zero}_raw.csv" \\
  --summary-json "$STAGE/{zero}_run.json" \\
  --layer {layer} \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-candidates {max_candidates} \\
  --scale {scale}

echo '--- Stage4-048 middle dense {pack} L{layer} {group} grouped restore ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{cand}.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{grouped}_raw.csv" \\
  --summary-json "$STAGE/{grouped}_run.json" \\
  --layer {layer} \\
  --top-ks 1,2,4,8,16,32,64,128,256 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-prompt-runs {max_prompt_runs}
"""


def _remote_script(args, layers: list[int], groups: list[str]) -> str:
    blocks = "\n".join(
        _block(
            args.pack,
            args.mode,
            layer,
            group,
            args.max_prompt_runs,
            args.candidate_pool_size,
            args.top_per_prompt_run,
            args.main_per_prompt_run,
            args.main_cap,
            args.max_candidates,
            args.scale,
        )
        for layer in layers
        for group in groups
    )
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
echo '--- Stage4-048 middle dense disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_evidence_first_feature_discovery.py \\
  scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py
if [ -z "$QWEN_MODEL" ]; then echo 'QWEN_MODEL_CACHE_MISSING'; exit 3; fi
{blocks}
"""


def _fetch(sftp, pack: str, mode: str, layers: list[int], groups: list[str]) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    fetched = 0
    missing = 0
    for layer in layers:
        for group in groups:
            for name in [
                f"{_stem(pack, mode, layer, group, 'candidates')}.csv",
                f"{_stem(pack, mode, layer, group, 'candidates')}.json",
                f"{_stem(pack, mode, layer, group, 'zeroing')}_raw.csv",
                f"{_stem(pack, mode, layer, group, 'zeroing')}_run.json",
                f"{_stem(pack, mode, layer, group, 'group')}_raw.csv",
                f"{_stem(pack, mode, layer, group, 'group')}_run.json",
            ]:
                try:
                    sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                    fetched += 1
                except FileNotFoundError:
                    missing += 1
    print(f"fetch complete: fetched={fetched} missing={missing}", flush=True)


def _status_command(pack: str, mode: str) -> str:
    return f"""
STAGE={REMOTE_STAGE}
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args --cols 260 | grep -E 'stage4_qwen_middle_dense|run_stage4_qwen_evidence_first|run_stage4_qwen_causal_cutter' | grep -v grep || true
echo FILES
ls -lh "$STAGE"/{PREFIX}_{pack}_{mode}_L* 2>/dev/null | tail -120 || true
echo LOGS
ls -lh "$STAGE"/logs/* 2>/dev/null | tail -40 || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-048 Qwen middle-layer dense PLT scan.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full", "strict-confirm"], default="smoke")
    parser.add_argument("--layers", default="10,13,14,15")
    parser.add_argument("--position-groups", default="all")
    parser.add_argument("--candidate-pool-size", type=int, default=32768)
    parser.add_argument("--top-per-prompt-run", type=int, default=64)
    parser.add_argument("--main-per-prompt-run", type=int, default=32)
    parser.add_argument("--main-cap", type=int, default=1024)
    parser.add_argument("--max-prompt-runs", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "full":
        if args.layers == "10,13,14,15":
            args.layers = "middle"
        if args.max_prompt_runs == 6:
            args.max_prompt_runs = 0
        if args.max_candidates == 6:
            args.max_candidates = 0
    layers = _parse_ints(args.layers)
    groups = _parse_groups(args.position_groups)

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.pack, args.mode))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err)
        client.close()
        return 0
    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch(sftp, args.pack, args.mode, layers, groups)
        sftp.close()
        client.close()
        return 0
    for script in [
        "run_stage4_qwen_evidence_first_feature_discovery.py",
        "run_stage4_qwen_evidence_first_intervention.py",
        "run_stage4_qwen_causal_cutter_validation.py",
        "run_cross_model_hidden_position_patch_smoke.py",
    ]:
        _upload_research_script(base, sftp, script)
    runs_path = PRIMARY_RUNS if args.pack == "primary" else STRICT_RUNS
    _upload_prompt_runs(base, sftp, runs_path, f"paperpack72_{args.pack}_prompt_runs.csv")
    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_middle_dense_{args.pack}_{args.mode}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args, layers, groups).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = f"{REMOTE_STAGE}/logs"
        log = f"{log_dir}/middle_dense_{args.pack}_{args.mode}_{stamp}.log"
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
    _fetch(sftp, args.pack, args.mode, layers, groups)
    sftp.close()
    client.close()
    if timed_out:
        raise TimeoutError("middle dense remote command timed out")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
