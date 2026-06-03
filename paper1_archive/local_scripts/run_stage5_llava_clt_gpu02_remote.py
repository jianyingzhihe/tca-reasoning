#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WSL_DISTRO = "Ubuntu2404"
REMOTE_HOST = "gcs-gpu02-wsl"
REMOTE_ROOT = "/home/xtyu/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/home/xtyu/stage5_clt_heterogeneity"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
REMOTE_PAPERPACK = f"{REMOTE_STAGE}/paperpack72"
REMOTE_LOG_DIR = f"{REMOTE_STAGE}/logs"
LOCAL_REPO = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm"
LOCAL_ASSETS = ROOT / "annotation" / "stage3_paperpack72_labelme" / "paperpack81_final_assets"
LOCAL_PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage5" / "cross_model"
PREFIX = "stage5_clt_heterogeneity"
ASSET_LABEL = "llava15_clt"
TRANSCODER_REF = "KokosDev/llava15-7b-clt"
LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_LAYERS = [0, 12, 15, 18, 21, 30]
DEFAULT_SMOKE_TOPKS = [1, 8, 32]
DEFAULT_FULL_TOPKS = [1, 4, 8, 16, 32, 64]


def _run(cmd: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[gpu02] {printable}", flush=True)
    if input_text is not None:
        return subprocess.run(
            cmd,
            input=input_text.replace("\r\n", "\n").encode("utf-8"),
            check=check,
        )
    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = [part for part in resolved.parts[1:]]
    return "/mnt/" + drive + "/" + "/".join(parts).replace("\\", "/")


def _parse_ints(raw: str, default: list[int]) -> list[int]:
    if raw in {"default", ""}:
        return list(default)
    if raw == "all":
        return list(DEFAULT_LAYERS)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _pack_paths(pack: str) -> tuple[str, str]:
    if pack == "primary":
        return (
            f"{REMOTE_PAPERPACK}/paperpack72_primary_manifest.csv",
            f"{REMOTE_PAPERPACK}/paperpack72_primary_prompt_runs.csv",
        )
    if pack == "strict":
        return (
            f"{REMOTE_PAPERPACK}/paperpack72_strict_sensitivity_manifest.csv",
            f"{REMOTE_PAPERPACK}/paperpack72_strict_sensitivity_prompt_runs.csv",
        )
    raise ValueError(f"unknown pack: {pack}")


def _ensure_local_inputs() -> None:
    required = [
        LOCAL_REPO / "scripts" / "server" / "setup_env.sh",
        LOCAL_REPO / "scripts" / "research" / "run_stage2o_attribution_weighted_feature_bridge.py",
        LOCAL_REPO / "scripts" / "research" / "run_stage2o_cross_model_source_control_probe.py",
        LOCAL_ASSETS / "images",
        LOCAL_ASSETS / "exported_masks",
        LOCAL_PAPERPACK / "paperpack72_primary_manifest.csv",
        LOCAL_PAPERPACK / "paperpack72_primary_prompt_runs.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing local inputs:\n" + "\n".join(missing))


def _ssh(script: str) -> None:
    _run(
        [
            "wsl",
            "-d",
            WSL_DISTRO,
            "--",
            "ssh",
            "-o",
            "BatchMode=yes",
            REMOTE_HOST,
            "bash",
            "-s",
        ],
        input_text=script,
    )


def _rsync(local: Path, remote: str, *, delete: bool = False, excludes: list[str] | None = None) -> None:
    source = _wsl_path(local)
    if local.is_dir() and not source.endswith("/"):
        source += "/"
    cmd = ["wsl", "-d", WSL_DISTRO, "--", "rsync", "-az"]
    if delete:
        cmd.append("--delete")
    for pattern in excludes or []:
        cmd.extend(["--exclude", pattern])
    cmd.extend([source, f"{REMOTE_HOST}:{remote}"])
    _run(cmd)


def sync_inputs() -> None:
    _ensure_local_inputs()
    _ssh(
        f"""
set -euo pipefail
mkdir -p {shlex.quote(REMOTE_ROOT)} {shlex.quote(REMOTE_STAGE)} {shlex.quote(REMOTE_ASSETS)} {shlex.quote(REMOTE_PAPERPACK)} {shlex.quote(REMOTE_LOG_DIR)}
hostname
whoami
python3.11 --version
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
df -h /home || true
"""
    )
    _rsync(
        LOCAL_REPO,
        REMOTE_ROOT + "/",
        delete=True,
        excludes=[".git/", ".venv/", "__pycache__/", ".pytest_cache/", "*.pyc", ".env"],
    )
    _rsync(LOCAL_ASSETS, REMOTE_ASSETS + "/", delete=True)
    _rsync(LOCAL_PAPERPACK, REMOTE_PAPERPACK + "/", delete=True)


def _remote_run_script(pack: str, mode: str, layers: list[int], topks: list[int], resume: bool) -> str:
    sample_manifest, run_manifest = _pack_paths(pack)
    max_runs = "--max-runs 6" if mode == "smoke" else ""
    layer_csv = ",".join(str(layer) for layer in layers)
    setup_stamp = "$(date +%Y%m%d_%H%M%S)"
    chunks: list[str] = [
        f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={shlex.quote(REMOTE_ROOT)}
STAGE={shlex.quote(REMOTE_STAGE)}
ASSET_ROOT={shlex.quote(REMOTE_ASSETS)}
LOG_DIR={shlex.quote(REMOTE_LOG_DIR)}
mkdir -p "$STAGE" "$LOG_DIR"
cd "$ROOT"

export HF_HOME="${{HF_HOME:-/home/xtyu/.cache/huggingface}}"
export HUGGINGFACE_HUB_CACHE="${{HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-$HF_HOME/transformers}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export CIRCUIT_TRACER_DISABLE_REMOTE_DB=1

if [ ! -f "$ROOT/.env" ]; then
  cat > "$ROOT/.env" <<'ENVEOF'
HF_HOME=/home/xtyu/.cache/huggingface
HUGGINGFACE_HUB_CACHE=/home/xtyu/.cache/huggingface/hub
TRANSFORMERS_CACHE=/home/xtyu/.cache/huggingface/transformers
CIRCUIT_TRACER_DISABLE_REMOTE_DB=1
ENVEOF
fi

if [ -x "$ROOT/.venv/bin/python" ]; then
  if ! "$ROOT/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    mv "$ROOT/.venv" "$ROOT/.venv_incompatible_{setup_stamp}"
  fi
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo '[gpu02] creating Python 3.11 venv'
  PYTHON_BIN=python3.11 bash scripts/server/setup_env.sh "$ROOT/.venv"
fi

source scripts/server/dev.sh "$ROOT/.env" "$ROOT/.venv"
export PYTHONPATH="$ROOT:${{PYTHONPATH:-}}"

echo '--- GPU02 run context ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
hostname
whoami
pwd
df -h /home || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

echo '--- py_compile research scripts ---'
python -m py_compile \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py

echo '--- prefetch requested LLaVA CLT transcoders ---'
python - <<'PY'
from huggingface_hub import hf_hub_download
layers = [{", ".join(str(layer) for layer in layers)}]
for layer in layers:
    path = hf_hub_download(repo_id="{TRANSCODER_REF}", filename=f"transcoder_L{{layer}}.pt", local_files_only=False)
    print(f"transcoder_L{{layer}}.pt: {{path}}", flush=True)
PY

MODEL_NAME="${{LLAVA_MODEL:-{LLAVA_MODEL}}}"
SAMPLE_MANIFEST={shlex.quote(sample_manifest)}
RUN_MANIFEST={shlex.quote(run_manifest)}
echo "MODEL_NAME=$MODEL_NAME"
echo "SAMPLE_MANIFEST=$SAMPLE_MANIFEST"
echo "RUN_MANIFEST=$RUN_MANIFEST"
echo "LAYERS={layer_csv}"
echo "TOPKS={','.join(str(topk) for topk in topks)}"
"""
    ]
    for layer in layers:
        for topk in topks:
            stem = f"{PREFIX}_{ASSET_LABEL}_{pack}_{mode}_L{layer}_topK{topk}"
            feature_csv = f"$STAGE/{stem}_feature_union.csv"
            source_csv = f"$STAGE/{stem}_source_control.csv"
            skip_feature = f'[ -s "{feature_csv}" ]' if resume else "false"
            skip_source = f'[ -s "{source_csv}" ]' if resume else "false"
            chunks.append(
                f"""
echo '--- feature bridge {stem} ---'
if {skip_feature}; then
  echo 'resume_skip_feature {stem}'
else
  python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
    --model-family llava \\
    --model-name "$MODEL_NAME" \\
    --transcoder-ref {TRANSCODER_REF} \\
    --annotation-roots "$ASSET_ROOT" \\
    --work-dir "$STAGE/{stem}_feature_work" \\
    --sample-manifest "$SAMPLE_MANIFEST" \\
    --run-manifest "$RUN_MANIFEST" \\
    --layer {layer} \\
    --mask-condition union_mask \\
    --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
    --top-k-features {topk} \\
    --control-pool-size 2048 \\
    {max_runs} \\
    --out-json "$STAGE/{stem}_feature_union.json" \\
    --out-csv "{feature_csv}" || true
fi

echo '--- source-control {stem} ---'
if {skip_source}; then
  echo 'resume_skip_source {stem}'
else
  python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
    --model-family llava \\
    --model-name "$MODEL_NAME" \\
    --transcoder-ref {TRANSCODER_REF} \\
    --annotation-roots "$ASSET_ROOT" \\
    --work-dir "$STAGE/{stem}_source_control_work" \\
    --sample-manifest "$SAMPLE_MANIFEST" \\
    --run-manifest "$RUN_MANIFEST" \\
    --layer {layer} \\
    --mask-conditions answer_mask,union_mask \\
    --position-group top_hidden_delta_plus_answer_adjacent \\
    --top-k-features {topk} \\
    --control-pool-size 2048 \\
    {max_runs} \\
    --out-json "$STAGE/{stem}_source_control.json" \\
    --out-csv "{source_csv}" || true
fi
"""
            )
    chunks.append(
        """
echo '--- GPU02 Stage5 LLaVA CLT run complete ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
"""
    )
    return "\n".join(chunks)


def _upload_and_launch(
    pack: str,
    mode: str,
    layers: list[int],
    topks: list[int],
    resume: bool,
    detach: bool,
    start_after_hhmm: str | None,
) -> None:
    sync_inputs()
    script = _remote_run_script(pack, mode, layers, topks, resume)
    remote_script = f"{REMOTE_STAGE}/run_stage5_llava_clt_gpu02_{pack}_{mode}.sh"
    _ssh(
        f"""
set -euo pipefail
mkdir -p {shlex.quote(REMOTE_STAGE)} {shlex.quote(REMOTE_LOG_DIR)}
cat > {shlex.quote(REMOTE_STAGE)}/run_stage5_llava_clt_gpu02_timegate.sh <<'GATEEOF'
#!/usr/bin/env bash
set -euo pipefail
TARGET_HHMM="${{START_AFTER_HHMM:-}}"
if [ -z "$TARGET_HHMM" ]; then
  exec "$@"
fi
if [ "${{#TARGET_HHMM}}" -ne 4 ]; then
  echo "[gpu02] invalid START_AFTER_HHMM=$TARGET_HHMM" >&2
  exit 2
fi
current_hh=$(date +%H)
current_mm=$(date +%M)
current_minutes=$((10#$current_hh * 60 + 10#$current_mm))
target_hh=${{TARGET_HHMM:0:2}}
target_mm=${{TARGET_HHMM:2:2}}
target_minutes=$((10#$target_hh * 60 + 10#$target_mm))
if [ "$current_minutes" -lt "$target_minutes" ]; then
  sleep_minutes=$((target_minutes - current_minutes))
else
  sleep_minutes=$((1440 - current_minutes + target_minutes))
fi
if [ "$sleep_minutes" -gt 0 ]; then
  echo "[gpu02] waiting $sleep_minutes minute(s) until $TARGET_HHMM local time before launch"
  sleep "$((sleep_minutes * 60))"
fi
exec "$@"
GATEEOF
chmod +x {shlex.quote(REMOTE_STAGE)}/run_stage5_llava_clt_gpu02_timegate.sh
cat > {shlex.quote(remote_script)} <<'SCRIPTEOF'
{script}
SCRIPTEOF
chmod +x {shlex.quote(remote_script)}
"""
    )
    if start_after_hhmm:
        gate_env = f"START_AFTER_HHMM={shlex.quote(start_after_hhmm)}"
        launch_cmd = f"bash {shlex.quote(REMOTE_STAGE)}/run_stage5_llava_clt_gpu02_timegate.sh bash {shlex.quote(remote_script)}"
    else:
        gate_env = ""
        launch_cmd = f"bash {shlex.quote(remote_script)}"
    if detach:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = f"{REMOTE_LOG_DIR}/stage5_llava_clt_gpu02_{pack}_{mode}_{stamp}.log"
        _ssh(
            f"""
set -euo pipefail
{gate_env} nohup {launch_cmd} > {shlex.quote(log)} 2>&1 < /dev/null &
echo '{{"detached_remote_pid":"'$!'", "remote_log":"{log}", "remote_script":"{remote_script}"}}'
"""
        )
    else:
        if start_after_hhmm:
            _ssh(f"START_AFTER_HHMM={shlex.quote(start_after_hhmm)} bash {shlex.quote(REMOTE_STAGE)}/run_stage5_llava_clt_gpu02_timegate.sh bash {shlex.quote(remote_script)}")
        else:
            _ssh(f"bash {shlex.quote(remote_script)}")


def status(pack: str, mode: str) -> None:
    _ssh(
        f"""
set -euo pipefail
STAGE={shlex.quote(REMOTE_STAGE)}
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args --cols 260 | grep -E 'stage5_llava_clt_gpu02|stage2o_attribution|stage2o_cross_model|setup_env|pip install|LlavaForConditionalGeneration' | grep -v grep || true
echo FILES
ls -lh "$STAGE"/{PREFIX}_{ASSET_LABEL}_{pack}_{mode}_L* 2>/dev/null | tail -120 || true
echo LOGS
ls -lh "$STAGE"/logs/* 2>/dev/null | tail -40 || true
echo LAST_LOG_TAIL
last_log=$(ls -t "$STAGE"/logs/stage5_llava_clt_gpu02_{pack}_{mode}_*.log 2>/dev/null | head -n 1 || true)
if [ -n "$last_log" ]; then
  echo "$last_log"
  tail -n 80 "$last_log" || true
fi
echo GPU
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
"""
    )


def fetch(pack: str, mode: str) -> None:
    LOCAL_CROSS.mkdir(parents=True, exist_ok=True)
    local = _wsl_path(LOCAL_CROSS)
    cmd = [
        "wsl",
        "-d",
        WSL_DISTRO,
        "--",
        "bash",
        "-lc",
        "mkdir -p "
        + shlex.quote(local)
        + " && rsync -az --ignore-missing-args "
        + shlex.quote(f"{REMOTE_HOST}:{REMOTE_STAGE}/{PREFIX}_{ASSET_LABEL}_{pack}_{mode}_*")
        + " "
        + shlex.quote(local + "/"),
    ]
    _run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage5 LLaVA-CLT on gcs-gpu02 via low-frequency WSL SSH.")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["smoke", "screen", "full", "strict-confirm"], default="smoke")
    parser.add_argument("--layers", default="default")
    parser.add_argument("--topks", default="")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-after-hhmm", default="")
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()

    topk_default = DEFAULT_SMOKE_TOPKS if args.mode == "smoke" else DEFAULT_FULL_TOPKS
    layers = _parse_ints(args.layers, DEFAULT_LAYERS)
    topks = _parse_ints(args.topks or "default", topk_default)

    if args.status:
        status(args.pack, args.mode)
        return 0
    if args.fetch_only:
        fetch(args.pack, args.mode)
        return 0
    if args.sync_only:
        sync_inputs()
        print(json.dumps({"status": "synced", "remote_root": REMOTE_ROOT, "remote_stage": REMOTE_STAGE}, indent=2))
        return 0

    start_after_hhmm = args.start_after_hhmm.strip() or None
    _upload_and_launch(args.pack, args.mode, layers, topks, args.resume, args.detach, start_after_hhmm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
