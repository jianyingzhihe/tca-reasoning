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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage6_unified_prompt_text"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage6_unified_routeidentity"


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


def _manifest_for(tag: str) -> Path:
    return LOCAL_CROSS / f"{PREFIX}_{tag}_manifest.csv"


def _parse_layers(raw: str, manifest: Path, mode: str) -> list[int]:
    rows = _filter_rows(_read_csv(manifest), mode)
    if raw in {"all", "selected"}:
        layers = sorted({int(float(row["layer"])) for row in rows if row.get("layer")})
        if not layers:
            raise ValueError(f"no layers found in manifest: {manifest}")
        return layers
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _filter_rows(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    key = "stage6_include_smoke" if mode == "smoke" else "stage6_include_full"
    return [row for row in rows if row.get(key, "1") == "1"]


def _artifact_prefix(mode: str, layer: int, tag: str) -> str:
    return f"{PREFIX}_{mode}_L{layer}_{tag}"


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


def _upload_assets(base, sftp, manifest: Path, remote_manifest_name: str) -> None:
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/exported_masks")
    base._put_file(sftp, manifest, f"{REMOTE_STAGE}/{remote_manifest_name}")
    rows = _read_csv(manifest)
    seen: set[str] = set()
    uploaded = 0
    skipped = 0
    for row in rows:
        image = Path(row.get("local_image_path", ""))
        image_name = Path(row.get("image_filename", "")).name
        if image.exists() and image_name:
            remote_image = f"{REMOTE_ASSETS}/images/{image_name}"
            if remote_image not in seen:
                seen.add(remote_image)
                if _put_if_missing(base, sftp, image, remote_image):
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
            mask = Path(row.get(key, ""))
            if mask.exists() and stem:
                remote_mask = f"{REMOTE_ASSETS}/exported_masks/{stem}/{out_name}"
                if remote_mask in seen:
                    continue
                seen.add(remote_mask)
                if _put_if_missing(base, sftp, mask, remote_mask):
                    uploaded += 1
                else:
                    skipped += 1
    print(f"uploaded Qwen routeidentity assets: rows={len(rows)} paths={len(seen)} uploaded={uploaded} skipped={skipped}", flush=True)


def _remote_script(
    mode: str,
    layers: list[int],
    tag: str,
    remote_manifest_name: str,
    smoke_candidates: int,
    full_candidates: int,
    max_total_candidates: int,
    scale: float,
) -> str:
    layers_py = ",".join(str(layer) for layer in layers)
    filtered_manifest = f"{PREFIX}_{mode}_{tag}_manifest.csv"
    blocks = []
    for layer in layers:
        artifact = _artifact_prefix(mode, layer, tag)
        layer_manifest = f"{artifact}_manifest.csv"
        blocks.append(
            f"""
echo '--- Stage6 unified Qwen routeidentity {mode} L{layer} ---'
.venv/bin/python - <<PY
import csv
from pathlib import Path
src = Path("$STAGE/{filtered_manifest}")
dst = Path("$STAGE/{layer_manifest}")
rows = list(csv.DictReader(src.open("r", encoding="utf-8-sig", newline="")))
rows = [row for row in rows if str(int(float(row.get("layer", "-999")))) == "{layer}"]
if rows:
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
else:
    dst.write_text("", encoding="utf-8")
print(f"layer_manifest={{dst}} rows={{len(rows)}}")
PY
if [ -s "$STAGE/{layer_manifest}" ]; then
  .venv/bin/python -u scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
    --model-name "$QWEN_MODEL" \\
    --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
    --candidate-manifest "$STAGE/{layer_manifest}" \\
    --selection all \\
    --image-root "$ASSET_ROOT/images" \\
    --mask-root "$ASSET_ROOT/exported_masks" \\
    --work-dir "$STAGE/work_{artifact}" \\
    --out-csv "$STAGE/{artifact}_raw.csv" \\
    --summary-json "$STAGE/{artifact}_run.json" \\
    --layer {layer} \\
    --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
    --max-candidates 0 \\
    --scale {scale}
else
  echo 'skip L{layer}: no candidates'
fi
"""
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

echo '--- Stage6 unified Qwen routeidentity preflight ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile scripts/research/run_stage4_qwen_causal_cutter_validation.py
if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING'
  exit 3
fi

.venv/bin/python - <<PY
import csv
from pathlib import Path
src = Path("$STAGE/{remote_manifest_name}")
dst = Path("$STAGE/{filtered_manifest}")
rows = list(csv.DictReader(src.open("r", encoding="utf-8-sig", newline="")))
allowed = {{{layers_py}}}
key = "stage6_include_smoke" if "{mode}" == "smoke" else "stage6_include_full"
rows = [row for row in rows if row.get(key, "1") == "1" and int(float(row.get("layer", "-999"))) in allowed]
rows.sort(key=lambda row: (
    row.get("sample_id", ""),
    row.get("stage6_question_variant", ""),
    row.get("stage6_prompt_family", ""),
    int(float(row.get("layer", 999))),
    int(float(row.get("candidate_rank_global", 999999))),
))
if "{mode}" == "smoke":
    grouped = {{}}
    for row in rows:
        key2 = (row.get("sample_id", ""), row.get("stage6_question_variant", ""), row.get("stage6_prompt_family", ""), row.get("layer", ""))
        grouped.setdefault(key2, []).append(row)
    picked = []
    for key2 in sorted(grouped):
        picked.extend(grouped[key2][: max(1, {smoke_candidates})])
    rows = picked
elif {full_candidates} > 0:
    grouped = {{}}
    for row in rows:
        key2 = (row.get("sample_id", ""), row.get("stage6_question_variant", ""), row.get("stage6_prompt_family", ""), row.get("layer", ""))
        grouped.setdefault(key2, []).append(row)
    picked = []
    for key2 in sorted(grouped):
        picked.extend(grouped[key2][: {full_candidates}])
    rows = picked
if {max_total_candidates} > 0:
    rows = rows[:{max_total_candidates}]
if rows:
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
else:
    dst.write_text("", encoding="utf-8")
print(f"filtered_manifest={{dst}} rows={{len(rows)}} mode={mode} smoke_candidates={smoke_candidates} full_candidates={full_candidates} max_total_candidates={max_total_candidates}")
PY

{''.join(blocks)}
echo '--- Stage6 unified Qwen routeidentity done ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true
"""


def _fetch(sftp, mode: str, layers: list[int], tag: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    fetched = 0
    missing = 0
    for layer in layers:
        artifact = _artifact_prefix(mode, layer, tag)
        for name in [f"{artifact}_manifest.csv", f"{artifact}_raw.csv", f"{artifact}_run.json"]:
            try:
                sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                fetched += 1
                print(f"fetched {name}", flush=True)
            except FileNotFoundError:
                missing += 1
    print(f"fetch complete: fetched={fetched} missing={missing}", flush=True)


def _status_command(mode: str, layers: list[int], tag: str) -> str:
    patterns = "|".join(_artifact_prefix(mode, layer, tag) for layer in layers)
    return f"""
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'stage6_unified_prompt_text|run_stage4_qwen_causal_cutter_validation.py|{patterns}' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{PREFIX}_{mode}_*_{tag}_* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*routeidentity* 2>/dev/null || true
tail -n 80 {REMOTE_STAGE}/logs/*routeidentity* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo DISK
df -h /root/autodl-tmp
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage6 unified Qwen route-identity rediscovery.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--tag", default="unified_v1")
    parser.add_argument("--smoke-candidates-per-group", type=int, default=2)
    parser.add_argument("--full-candidates-per-group", type=int, default=0)
    parser.add_argument("--max-total-candidates", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    manifest = _manifest_for(args.tag)
    if not manifest.exists() and not args.status and not args.fetch_only:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "local" / "build_stage6_unified_metric_manifests.py"), "--tag", args.tag],
            check=True,
        )
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    layers = _parse_layers(args.layers, manifest, args.mode)
    remote_manifest_name = f"{PREFIX}_{args.tag}_manifest.csv"

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)

    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.mode, layers, args.tag))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch(sftp, args.mode, layers, args.tag)
        sftp.close()
        client.close()
        if not args.skip_analyze:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "local" / "analyze_stage6_unified_prompt_text_metrics.py"),
                    "--tag",
                    args.tag,
                    "--mode",
                    args.mode,
                ],
                check=False,
            )
        return 0

    _upload_research_script(base, sftp, "run_stage4_qwen_causal_cutter_validation.py")
    _upload_assets(base, sftp, manifest, remote_manifest_name)

    remote_script = f"{REMOTE_STAGE}/run_stage6_unified_routeidentity_{args.mode}_{args.tag}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                mode=args.mode,
                layers=layers,
                tag=args.tag,
                remote_manifest_name=remote_manifest_name,
                smoke_candidates=args.smoke_candidates_per_group,
                full_candidates=max(0, args.full_candidates_per_group),
                max_total_candidates=max(0, args.max_total_candidates),
                scale=args.scale,
            ).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        remote_log_dir = f"{REMOTE_STAGE}/logs"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log = f"{remote_log_dir}/routeidentity_{args.mode}_{stamp}.log"
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
    exit_status = 124 if timed_out else stdout.channel.recv_exit_status()
    for stream, is_err in [(stdout, False), (stderr, True)]:
        try:
            data = stream.read()
            if data:
                print(data.decode("utf-8", errors="replace"), end="", file=sys.stderr if is_err else sys.stdout)
        except Exception:
            pass
    sftp = client.open_sftp()
    _fetch(sftp, args.mode, layers, args.tag)
    sftp.close()
    client.close()
    if exit_status == 0 and not args.skip_analyze:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage6_unified_prompt_text_metrics.py"),
                "--tag",
                args.tag,
                "--mode",
                args.mode,
            ],
            check=False,
        )
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
