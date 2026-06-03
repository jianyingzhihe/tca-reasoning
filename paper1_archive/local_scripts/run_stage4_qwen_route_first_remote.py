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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_route_first"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PREFIX = "stage4_qwen_route_first"

BASE_MANIFESTS = {
    "primary": LOCAL_CROSS / "stage4_qwen_route_first_primary_manifest.csv",
    "strict": LOCAL_CROSS / "stage4_qwen_route_first_strict_manifest.csv",
}


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


def _manifest_for(pack: str, selection: str, tag: str, manifest_path: str = "") -> Path:
    if manifest_path:
        return Path(manifest_path)
    if selection == "frozen":
        if not tag:
            raise ValueError("--selection frozen requires --tag")
        return LOCAL_CROSS / f"{PREFIX}_strict_frozen_{tag}_manifest.csv"
    return BASE_MANIFESTS[pack]


def _parse_layers(raw: str, manifest: Path) -> list[int]:
    if raw in {"selected", "all"}:
        rows = _read_csv(manifest)
        layers = sorted({int(float(row["layer"])) for row in rows if row.get("layer")})
        if not layers:
            raise ValueError(f"no layers found in manifest: {manifest}")
        return layers
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _artifact_prefix(pack: str, mode: str, layer: int, tag: str) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"{PREFIX}_{pack}_{mode}_L{layer}{suffix}"


def _remote_asset_root(tag: str) -> str:
    if tag:
        return f"{REMOTE_STAGE}/assets_{tag}"
    return REMOTE_ASSETS


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


def _upload_assets(base, sftp, manifest: Path, remote_manifest_name: str, remote_assets: str) -> None:
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{remote_assets}/images")
    base._mkdir_p(sftp, f"{remote_assets}/exported_masks")
    base._put_file(sftp, manifest, f"{REMOTE_STAGE}/{remote_manifest_name}")
    rows = _read_csv(manifest)
    uploaded = 0
    skipped = 0
    seen_remotes: set[str] = set()
    for row in rows:
        image = Path(row.get("local_image_path", ""))
        image_name = Path(row.get("image_filename", "")).name
        if image.exists() and image_name:
            remote_image = f"{remote_assets}/images/{image_name}"
            if remote_image not in seen_remotes:
                seen_remotes.add(remote_image)
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
                remote_mask = f"{remote_assets}/exported_masks/{stem}/{out_name}"
                if remote_mask in seen_remotes:
                    continue
                seen_remotes.add(remote_mask)
                if _put_if_missing(base, sftp, mask, remote_mask):
                    uploaded += 1
                else:
                    skipped += 1
    print(
        f"uploaded route-first assets: rows={len(rows)} unique_remote_paths={len(seen_remotes)} uploaded={uploaded} skipped={skipped}",
        flush=True,
    )


def _remote_script(
    *,
    pack: str,
    mode: str,
    layers: list[int],
    tag: str,
    smoke_candidates: int,
    scale: float,
    selection: str,
    remote_manifest_name: str,
    remote_assets: str,
) -> str:
    layers_py = ",".join(str(layer) for layer in layers)
    suffix = f"_{tag}" if tag else ""
    filtered_manifest = f"{PREFIX}_{pack}_{mode}{suffix}_manifest.csv"
    layer_blocks = []
    for layer in layers:
        artifact = _artifact_prefix(pack, mode, layer, tag)
        layer_manifest = f"{artifact}_manifest.csv"
        layer_blocks.append(
            f"""
echo '--- Stage4-060 route-first {pack} {mode} L{layer} ---'
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
ASSET_ROOT={remote_assets}
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
mkdir -p "$STAGE"

echo '--- Stage4-060 Qwen route-first disk/gpu ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py \\
  scripts/research/run_cross_model_hidden_position_patch_smoke.py

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
selection = "{selection}"
rows = [row for row in rows if int(float(row.get("layer", "-999"))) in allowed]
if selection == "frozen":
    rows = [row for row in rows if row.get("include_frozen") == "1"]
else:
    rows = [row for row in rows if row.get("include_pool", "1") == "1"]
rows.sort(key=lambda row: int(float(row.get("candidate_rank_global") or row.get("candidate_id", "999999").split("_")[-1] or 999999)))
if "{mode}" == "smoke":
    grouped = {{}}
    for row in rows:
        grouped.setdefault(int(float(row.get("layer", -999))), []).append(row)
    picked = []
    while len(picked) < {smoke_candidates} and any(grouped.values()):
        for layer in sorted(grouped):
            if grouped[layer] and len(picked) < {smoke_candidates}:
                picked.append(grouped[layer].pop(0))
    rows = picked
if rows:
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
else:
    dst.write_text("", encoding="utf-8")
print(f"filtered_manifest={{dst}} rows={{len(rows)}} mode={mode} selection={selection}")
PY

{''.join(layer_blocks)}
"""


def _fetch(sftp, pack: str, mode: str, layers: list[int], tag: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    fetched = 0
    missing = 0
    for layer in layers:
        artifact = _artifact_prefix(pack, mode, layer, tag)
        for name in [f"{artifact}_manifest.csv", f"{artifact}_raw.csv", f"{artifact}_run.json"]:
            try:
                sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                fetched += 1
                print(f"fetched {name}", flush=True)
            except FileNotFoundError:
                missing += 1
    print(f"fetch complete: fetched={fetched} missing={missing}", flush=True)


def _status_command(pack: str, mode: str, layers: list[int], tag: str) -> str:
    patterns = "|".join(_artifact_prefix(pack, mode, layer, tag) for layer in layers)
    return f"""
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'stage4_qwen_route_first|run_stage4_qwen_causal_cutter_validation.py|{patterns}' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{PREFIX}_{pack}_{mode}_* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*route_first* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-060 Qwen route-first causal node validation.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--layers", default="selected")
    parser.add_argument("--selection", choices=["all", "frozen"], default="all")
    parser.add_argument("--tag", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--smoke-candidates", type=int, default=12)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    manifest = _manifest_for(args.pack, args.selection, args.tag, args.manifest_path)
    if not manifest.exists() and not args.status and not args.fetch_only:
        build_cmd = [sys.executable, str(ROOT / "scripts" / "local" / "build_stage4_qwen_route_first_manifest.py")]
        if args.selection == "frozen":
            build_cmd.extend(["--strict-from-primary", args.tag])
        subprocess.run(build_cmd, check=True)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    layers = _parse_layers(args.layers, manifest)
    remote_manifest_name = f"{args.pack}_{args.selection}{('_' + args.tag) if args.tag else ''}_manifest.csv"

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)

    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.pack, args.mode, layers, args.tag))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch(sftp, args.pack, args.mode, layers, args.tag)
        sftp.close()
        client.close()
        if not args.skip_analyze:
            analyze_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_route_first_validation.py"),
                "--pack",
                args.pack,
                "--mode",
                args.mode,
            ]
            if args.tag:
                analyze_cmd.extend(["--tag", args.tag])
            subprocess.run(analyze_cmd, check=False)
        return 0

    for script_name in [
        "run_stage4_qwen_causal_cutter_validation.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    remote_assets = _remote_asset_root(args.tag)
    _upload_assets(base, sftp, manifest, remote_manifest_name, remote_assets)

    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_route_first_{args.pack}_{args.mode}{('_' + args.tag) if args.tag else ''}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                pack=args.pack,
                mode=args.mode,
                layers=layers,
                tag=args.tag,
                smoke_candidates=args.smoke_candidates,
                scale=args.scale,
                selection=args.selection,
                remote_manifest_name=remote_manifest_name,
                remote_assets=remote_assets,
            ).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        remote_log_dir = f"{REMOTE_STAGE}/logs"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log = f"{remote_log_dir}/route_first_{args.pack}_{args.mode}_{stamp}.log"
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
    _fetch(sftp, args.pack, args.mode, layers, args.tag)
    sftp.close()
    client.close()
    if not args.skip_analyze:
        analyze_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_route_first_validation.py"),
            "--pack",
            args.pack,
            "--mode",
            args.mode,
        ]
        if args.tag:
            analyze_cmd.extend(["--tag", args.tag])
        subprocess.run(analyze_cmd, check=False)
    if exit_status != 0:
        raise SystemExit(exit_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
