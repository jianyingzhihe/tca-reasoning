#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _f(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _parse_layers(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _auto_layers(limit: int) -> list[int]:
    spec = _read_csv(LOCAL_CROSS / "stage4_qwen_decisive_route_specificity.csv")
    layer_scores: dict[int, float] = {}
    for row in spec:
        if row.get("pack") != "primary" or row.get("stage") != "hidden_lattice":
            continue
        metric = row.get("metric")
        if metric not in {"hidden_effect", "hidden_real_minus_shifted", "hidden_real_minus_shuffled", "hidden_correct_minus_wrong"}:
            continue
        layer = int(float(row.get("layer", "0")))
        weight = 2.0 if metric == "hidden_correct_minus_wrong" else 1.0
        score = weight * (_f(row.get("ci95_low")) + 0.25 * _f(row.get("mean")) + 0.1 * _f(row.get("positive_frac")))
        layer_scores[layer] = layer_scores.get(layer, 0.0) + score
    out = [layer for layer, _score in sorted(layer_scores.items(), key=lambda item: item[1], reverse=True)[:limit]]
    return out or [16, 20, 24, 26]


def _stem(pack: str, mode: str, layer: int, artifact: str) -> str:
    return f"stage4_qwen_decisive_route_plt_{pack}_{mode}_L{layer}_{artifact}"


def _pack_layer_block(
    pack: str,
    mode: str,
    layer: int,
    max_prompt_runs: int,
    top_per_prompt_run: int,
    main_per_prompt_run: int,
    max_clean_rank: int,
) -> str:
    runs_name = f"paperpack72_{pack}_prompt_runs.csv"
    cand = _stem(pack, mode, layer, "candidates")
    zero = _stem(pack, mode, layer, "zeroing")
    group = _stem(pack, mode, layer, "group")
    max_candidates = 6 if mode == "smoke" else 0
    return f"""
echo '--- Stage4-024 PLT layer sweep {pack} L{layer} discovery ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_feature_discovery.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --prompt-runs "$STAGE/{runs_name}" \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{cand}.csv" \\
  --summary-json "$STAGE/{cand}.json" \\
  --layer {layer} \\
  --position-group visual_answer \\
  --candidate-pool-size 8192 \\
  --top-per-prompt-run {top_per_prompt_run} \\
  --main-per-prompt-run {main_per_prompt_run} \\
  --max-clean-rank {max_clean_rank} \\
  --max-prompt-runs {max_prompt_runs}

echo '--- Stage4-024 PLT layer sweep {pack} L{layer} zeroing controls ---'
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
  --max-candidates {max_candidates}

echo '--- Stage4-024 PLT layer sweep {pack} L{layer} grouped restore ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{cand}.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{group}_raw.csv" \\
  --summary-json "$STAGE/{group}_run.json" \\
  --layer {layer} \\
  --top-ks 1,4,8,16,32,64 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-prompt-runs {max_prompt_runs}
"""


def _remote_script(
    mode: str,
    packs: list[str],
    layers: list[int],
    max_prompt_runs: int,
    top_per_prompt_run: int,
    main_per_prompt_run: int,
    max_clean_rank: int,
) -> str:
    blocks = "\n".join(
        _pack_layer_block(pack, mode, layer, max_prompt_runs, top_per_prompt_run, main_per_prompt_run, max_clean_rank)
        for pack in packs
        for layer in layers
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

echo '--- Stage4-024 Qwen PLT layer sweep disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_evidence_first_feature_discovery.py \\
  scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: PLT layer sweep skipped'
  exit 3
fi

{blocks}
"""


def _fetch(sftp, mode: str, packs: list[str], layers: list[int]) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    for pack in packs:
        for layer in layers:
            names = [
                f"{_stem(pack, mode, layer, 'candidates')}.csv",
                f"{_stem(pack, mode, layer, 'candidates')}.json",
                f"{_stem(pack, mode, layer, 'zeroing')}_raw.csv",
                f"{_stem(pack, mode, layer, 'zeroing')}_run.json",
                f"{_stem(pack, mode, layer, 'group')}_raw.csv",
                f"{_stem(pack, mode, layer, 'group')}_run.json",
            ]
            for name in names:
                try:
                    sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                    print(f"fetched {name}", flush=True)
                except FileNotFoundError:
                    print(f"missing {name}", flush=True)


def _status_command(mode: str, packs: list[str], layers: list[int]) -> str:
    patterns = "|".join(f"stage4_qwen_decisive_route_plt_{pack}_{mode}_L{layer}" for pack in packs for layer in layers)
    return f"""
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'run_stage4_qwen_evidence_first|run_stage4_qwen_causal_cutter|{patterns}' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/stage4_qwen_decisive_route_plt_*_{mode}_L* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*plt_layer_sweep* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-024 Qwen PLT layer-matched feature sweep on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--packs", default="primary")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--auto-layer-limit", type=int, default=3)
    parser.add_argument("--max-prompt-runs", type=int, default=6)
    parser.add_argument("--top-per-prompt-run", type=int, default=4)
    parser.add_argument("--main-per-prompt-run", type=int, default=2)
    parser.add_argument("--max-clean-rank", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Launch remote job under nohup and return immediately.")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch expected artifacts; do not upload or start a job.")
    parser.add_argument("--status", action="store_true", help="Print remote process/file status for this mode/packs/layers.")
    args = parser.parse_args()
    packs = [part.strip() for part in args.packs.split(",") if part.strip()]
    if args.mode == "full" and args.max_prompt_runs == 6:
        args.max_prompt_runs = 0
    layers = _auto_layers(args.auto_layer_limit) if args.layers == "auto" else _parse_layers(args.layers)
    layers = [layer for layer in layers if 0 <= layer <= 27]
    for pack in packs:
        if pack not in {"primary", "strict"}:
            raise ValueError(f"unknown pack: {pack}")
    if not layers:
        decision = {"status": "blocked", "reason": "no_valid_layers", "mode": args.mode}
        _write_json(LOCAL_CROSS / "stage4_qwen_decisive_route_plt_layer_sweep_decision.json", decision)
        print(json.dumps(decision, indent=2, ensure_ascii=False))
        return 2

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.mode, packs, layers))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err)
        client.close()
        return 0
    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch(sftp, args.mode, packs, layers)
        sftp.close()
        client.close()
        return 0
    for script in [
        "run_stage4_qwen_evidence_first_feature_discovery.py",
        "run_stage4_qwen_evidence_first_intervention.py",
        "run_stage4_qwen_causal_cutter_validation.py",
        "run_stage4_qwen_evidence_linked_cutter_v2.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
    ]:
        _upload_research_script(base, sftp, script)
    if "primary" in packs:
        _upload_prompt_runs(base, sftp, PRIMARY_RUNS, "paperpack72_primary_prompt_runs.csv")
    if "strict" in packs:
        _upload_prompt_runs(base, sftp, STRICT_RUNS, "paperpack72_strict_prompt_runs.csv")
    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_plt_layer_sweep_{args.mode}_{'_'.join(packs)}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                args.mode,
                packs,
                layers,
                args.max_prompt_runs,
                args.top_per_prompt_run,
                args.main_per_prompt_run,
                args.max_clean_rank,
            ).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log_dir = f"{REMOTE_STAGE}/logs"
        layers_label = "_".join(str(layer) for layer in layers)
        remote_log = f"{remote_log_dir}/plt_layer_sweep_{args.mode}_{'_'.join(packs)}_L{layers_label}_{stamp}.log"
        detach_cmd = f"mkdir -p {remote_log_dir}; nohup bash {remote_script} > {remote_log} 2>&1 < /dev/null & echo $!"
        _stdin, stdout, stderr = client.exec_command(detach_cmd)
        pid = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        print(json.dumps({"detached_remote_pid": pid, "remote_log": remote_log, "remote_script": remote_script}, indent=2), flush=True)
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
    _fetch(sftp, args.mode, packs, layers)
    sftp.close()
    client.close()

    decision = {
        "status": "completed" if exit_status == 0 else "blocked",
        "mode": args.mode,
        "packs": packs,
        "layers": layers,
        "exit_status": exit_status,
        "claim_boundary": "Layer sweep artifacts require analyze_stage4_qwen_decisive_route.py before scientific interpretation.",
    }
    _write_json(LOCAL_CROSS / "stage4_qwen_decisive_route_plt_layer_sweep_decision.json", decision)
    if not args.skip_analyze and exit_status == 0:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_decisive_route.py"), "--mode", args.mode], check=False)
    if timed_out:
        raise TimeoutError("remote Qwen PLT layer sweep exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
