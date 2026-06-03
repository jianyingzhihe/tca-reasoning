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
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_qwen_evidence_first"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
PRIMARY_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
STRICT_RUNS = PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"
PRIMARY_SOURCE_TRACING = LOCAL_CROSS / "stage4_qwen_source_tracing_primary_full_expanded_v2_L26_top32_intervention.csv"
STRICT_SOURCE_TRACING = LOCAL_CROSS / "stage4_qwen_source_tracing_strict_full_expanded_v2_L26_top32_intervention.csv"


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


def _mask_path(row: dict[str, str], name: str) -> Path:
    if row.get("mask_dir"):
        return Path(row["mask_dir"]) / name
    return Path("")


def _upload_prompt_run_assets(base, sftp, runs_path: Path, remote_name: str) -> None:
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
        for local_name, remote_name_mask in [
            ("answer.png", "answer.png"),
            ("union.png", "union.png"),
            ("shifted.png", "shifted.png"),
            ("shuffled.png", "shuffled.png"),
        ]:
            key = (stem, remote_name_mask)
            if key in seen_masks:
                continue
            seen_masks.add(key)
            local_mask = _mask_path(row, local_name)
            if not local_mask.exists():
                continue
            if _put_if_missing(base, sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{remote_name_mask}"):
                uploaded += 1
            else:
                skipped += 1
    print(f"uploaded {remote_name}: rows={len(rows)} uploaded={uploaded} skipped_existing={skipped}", flush=True)


def _upload_source_tracing(base, sftp, pack: str) -> None:
    local = PRIMARY_SOURCE_TRACING if pack == "primary" else STRICT_SOURCE_TRACING
    if local.exists():
        base._put_file(sftp, local, f"{REMOTE_STAGE}/stage4_qwen_source_tracing_{pack}_expanded_v2_intervention.csv")
        print(f"uploaded source tracing intervention for {pack}", flush=True)
    else:
        print(f"missing local source tracing intervention for {pack}: {local}", flush=True)


def _stem(source: str, pack: str, mode: str) -> str:
    return f"stage4_qwen_{source}_{pack}_{mode}"


def _pack_script(pack: str, mode: str, max_prompt_runs: int, run_adapter: bool, skip_discovery: bool) -> str:
    runs_name = f"paperpack72_{pack}_prompt_runs.csv"
    stem = _stem("evidence_first", pack, mode)
    adapter_stem = _stem("adapter_v3", pack, mode)
    max_candidates = 6 if mode == "smoke" else 0
    adapter_block = ""
    if run_adapter:
        adapter_block = f"""
.venv/bin/python -u scripts/research/run_stage4_qwen_adapter_v3_route_probe.py \\
  --source-tracing-intervention "$STAGE/stage4_qwen_source_tracing_{pack}_expanded_v2_intervention.csv" \\
  --evidence-discovery "$STAGE/{stem}_candidates.csv" \\
  --out-manifest "$STAGE/{adapter_stem}_manifest.csv" \\
  --summary-json "$STAGE/{adapter_stem}_manifest.json" \\
  --allow-feature-fallback

.venv/bin/python -u scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{adapter_stem}_manifest.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --work-dir "$STAGE/work_{adapter_stem}_zeroing" \\
  --out-csv "$STAGE/{adapter_stem}_zeroing_raw.csv" \\
  --summary-json "$STAGE/{adapter_stem}_zeroing_run.json" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-candidates {max_candidates}

.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{adapter_stem}_manifest.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{adapter_stem}_group_raw.csv" \\
  --summary-json "$STAGE/{adapter_stem}_group_run.json" \\
  --layer 26 \\
  --top-ks 1,4,8,16 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-prompt-runs {max_prompt_runs}
"""
    discovery_block = ""
    if not skip_discovery:
        discovery_block = f"""
echo '--- Stage4-020 {pack} evidence-first discovery ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_feature_discovery.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --prompt-runs "$STAGE/{runs_name}" \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{stem}_candidates.csv" \\
  --summary-json "$STAGE/{stem}_candidates.json" \\
  --layer 26 \\
  --position-group visual_answer \\
  --candidate-pool-size 8192 \\
  --top-per-prompt-run 4 \\
  --main-per-prompt-run 2 \\
  --max-prompt-runs {max_prompt_runs}
"""
    else:
        discovery_block = f"""
echo '--- Stage4-020 {pack} resume after discovery ---'
test -s "$STAGE/{stem}_candidates.csv"
"""

    return f"""
{discovery_block}

echo '--- Stage4-020 {pack} evidence-first clean/source controls ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{stem}_candidates.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --work-dir "$STAGE/work_{stem}_zeroing" \\
  --out-csv "$STAGE/{stem}_zeroing_raw.csv" \\
  --summary-json "$STAGE/{stem}_zeroing_run.json" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-candidates {max_candidates}

echo '--- Stage4-020 {pack} evidence-first grouped restore ---'
.venv/bin/python -u scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-plt \\
  --candidate-manifest "$STAGE/{stem}_candidates.csv" \\
  --selection main \\
  --image-root "$ASSET_ROOT/images" \\
  --mask-root "$ASSET_ROOT/exported_masks" \\
  --out-csv "$STAGE/{stem}_group_raw.csv" \\
  --summary-json "$STAGE/{stem}_group_run.json" \\
  --layer 26 \\
  --top-ks 1,4,8,16 \\
  --mask-conditions answer_mask,union_mask,shifted_mask,shuffled_mask \\
  --max-prompt-runs {max_prompt_runs}
{adapter_block}
"""


def _remote_script(mode: str, packs: list[str], max_prompt_runs: int, run_adapter: bool, skip_discovery: bool) -> str:
    pack_blocks = "\n".join(_pack_script(pack, mode, max_prompt_runs, run_adapter, skip_discovery) for pack in packs)
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

echo '--- Stage4-020 Qwen evidence-first disk/gpu ---'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile \\
  scripts/research/run_stage4_qwen_evidence_first_feature_discovery.py \\
  scripts/research/run_stage4_qwen_evidence_first_intervention.py \\
  scripts/research/run_stage4_qwen_adapter_v3_route_probe.py \\
  scripts/research/run_stage4_qwen_causal_cutter_validation.py \\
  scripts/research/run_stage4_qwen_evidence_linked_cutter_v2.py \\
  scripts/research/run_cross_model_feature_intervention_smoke.py

if [ -z "$QWEN_MODEL" ]; then
  echo 'QWEN_MODEL_CACHE_MISSING: Stage4-020 skipped'
  exit 3
fi

{pack_blocks}
"""


def _fetch(sftp, mode: str, packs: list[str], run_adapter: bool) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    suffixes = [
        "candidates.csv",
        "candidates.json",
        "zeroing_raw.csv",
        "zeroing_run.json",
        "group_raw.csv",
        "group_run.json",
    ]
    adapter_suffixes = [
        "manifest.csv",
        "manifest.json",
        "zeroing_raw.csv",
        "zeroing_run.json",
        "group_raw.csv",
        "group_run.json",
    ]
    for pack in packs:
        for suffix in suffixes:
            name = f"{_stem('evidence_first', pack, mode)}_{suffix}"
            try:
                sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                print(f"fetched {name}", flush=True)
            except FileNotFoundError:
                print(f"missing {name}", flush=True)
        if run_adapter:
            for suffix in adapter_suffixes:
                name = f"{_stem('adapter_v3', pack, mode)}_{suffix}"
                try:
                    sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
                    print(f"fetched {name}", flush=True)
                except FileNotFoundError:
                    print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage4-020 Qwen evidence-first route validation on AutoDL.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--packs", default="primary", help="Comma-separated: primary,strict")
    parser.add_argument("--max-prompt-runs", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--skip-adapter", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--resume-after-discovery", action="store_true", help="Reuse existing remote candidates and run zeroing/group/adapter stages only.")
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
    for script_name in [
        "run_stage4_qwen_evidence_first_feature_discovery.py",
        "run_stage4_qwen_evidence_first_intervention.py",
        "run_stage4_qwen_adapter_v3_route_probe.py",
        "run_stage4_qwen_causal_cutter_validation.py",
        "run_stage4_qwen_evidence_linked_cutter_v2.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    if "primary" in packs:
        _upload_prompt_run_assets(base, sftp, PRIMARY_RUNS, "paperpack72_primary_prompt_runs.csv")
        _upload_source_tracing(base, sftp, "primary")
    if "strict" in packs:
        _upload_prompt_run_assets(base, sftp, STRICT_RUNS, "paperpack72_strict_prompt_runs.csv")
        _upload_source_tracing(base, sftp, "strict")
    remote_script = f"{REMOTE_STAGE}/run_stage4_qwen_evidence_first_{args.mode}_{'_'.join(packs)}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.mode, packs, args.max_prompt_runs, not args.skip_adapter, args.resume_after_discovery).replace("\r\n", "\n"))
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
    _fetch(sftp, args.mode, packs, not args.skip_adapter)
    sftp.close()
    client.close()
    if not args.skip_analyze and exit_status == 0:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage4_qwen_evidence_first_route.py"),
                "--mode",
                args.mode,
            ],
            check=False,
        )
    if timed_out:
        raise TimeoutError("remote Stage4-020 Qwen evidence-first run exceeded timeout")
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
