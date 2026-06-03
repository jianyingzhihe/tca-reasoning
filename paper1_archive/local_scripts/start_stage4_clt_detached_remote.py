#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage4_clt_finalization"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"


ASSETS = {
    "qwen_clt": {
        "label": "qwen2p5vl_clt",
        "family": "qwen",
        "transcoder": "KokosDev/qwen2p5vl-7b-clt",
        "layers": [26],
        "model_expr": '"$QWEN_MODEL"',
        "require": '[ -n "$QWEN_MODEL" ]',
    },
    "llava_clt": {
        "label": "llava15_clt",
        "family": "llava",
        "transcoder": "KokosDev/llava15-7b-clt",
        "layers": [12, 15, 18, 21],
        "model_expr": '"$LLAVA_MODEL"',
        "require": '[ -n "$LLAVA_MODEL" ]',
    },
}


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


def _remote_script(asset: str, pack: str, mode: str, topks: list[int]) -> str:
    cfg = ASSETS[asset]
    label = cfg["label"]
    family = cfg["family"]
    transcoder = cfg["transcoder"]
    model_expr = cfg["model_expr"]
    require = cfg["require"]
    blocks: list[str] = []
    for layer in cfg["layers"]:
        for topk in topks:
            suffix = f"{label}_{pack}_{mode}_L{layer}_topK{topk}"
            blocks.append(
                f"""
if ! {require}; then
  echo 'STAGE4_CLT_ASSET_MISSING: {label}'
  exit 3
fi

if [ -s "$STAGE/stage4_{suffix}_feature_union.csv" ] && [ -s "$STAGE/stage4_{suffix}_feature_union.json" ]; then
  echo '[skip] feature exists: {suffix}'
else
  echo '[run] feature bridge: {suffix}'
  .venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
    --model-family {family} \\
    --model-name {model_expr} \\
    --transcoder-ref {transcoder} \\
    --annotation-roots "$ASSET_ROOT" \\
    --work-dir "$STAGE/{suffix}_feature_work" \\
    --sample-manifest "$SAMPLE_MANIFEST" \\
    --run-manifest "$RUN_MANIFEST" \\
    --layer {layer} \\
    --mask-condition union_mask \\
    --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
    --top-k-features {topk} \\
    --control-pool-size 2048 \\
    --out-json "$STAGE/stage4_{suffix}_feature_union.json" \\
    --out-csv "$STAGE/stage4_{suffix}_feature_union.csv"
fi

if [ -s "$STAGE/stage4_{suffix}_source_control.csv" ] && [ -s "$STAGE/stage4_{suffix}_source_control.json" ]; then
  echo '[skip] source-control exists: {suffix}'
else
  echo '[run] source-control: {suffix}'
  .venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
    --model-family {family} \\
    --model-name {model_expr} \\
    --transcoder-ref {transcoder} \\
    --annotation-roots "$ASSET_ROOT" \\
    --work-dir "$STAGE/{suffix}_source_control_work" \\
    --sample-manifest "$SAMPLE_MANIFEST" \\
    --run-manifest "$RUN_MANIFEST" \\
    --layer {layer} \\
    --mask-conditions answer_mask,union_mask \\
    --position-group top_hidden_delta_plus_answer_adjacent \\
    --top-k-features {topk} \\
    --control-pool-size 2048 \\
    --out-json "$STAGE/stage4_{suffix}_source_control.json" \\
    --out-csv "$STAGE/stage4_{suffix}_source_control.csv"
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
SAMPLE_MANIFEST="$STAGE/paperpack_{pack}_manifest.csv"
RUN_MANIFEST="$STAGE/paperpack_{pack}_prompt_runs.csv"
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
LLAVA_LOCAL=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf
LLAVA_HF=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/* 2>/dev/null | head -n 1 || true)
if [ -d "$LLAVA_LOCAL" ]; then
  LLAVA_MODEL="$LLAVA_LOCAL"
elif [ -n "$LLAVA_HF" ]; then
  LLAVA_MODEL="$LLAVA_HF"
else
  LLAVA_MODEL="llava-hf/llava-1.5-7b-hf"
fi

echo '--- detached Stage4 CLT finalization start ---'
date
df -h /root/autodl-tmp || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

{''.join(blocks)}

echo '--- detached Stage4 CLT finalization done ---'
date
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Start detached resumable Stage4 CLT finalization job.")
    parser.add_argument("--asset", choices=["qwen_clt", "llava_clt"], default="qwen_clt")
    parser.add_argument("--pack", choices=["primary", "strict"], default="primary")
    parser.add_argument("--mode", choices=["full"], default="full")
    parser.add_argument("--topks", default="4,8,16,32")
    args = parser.parse_args()
    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    for script_name in [
        "run_stage2o_attribution_weighted_feature_bridge.py",
        "run_stage2o_cross_model_source_control_probe.py",
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_cross_model_mask_shuffled_negative_control_smoke.py",
        "run_cross_model_wrong_target_negative_control_smoke.py",
    ]:
        _upload_research_script(base, sftp, script_name)
    suffix = f"{args.asset}_{args.pack}_{args.mode}_{'_'.join(str(x) for x in topks)}"
    remote_script = f"{REMOTE_STAGE}/detached_stage4_clt_{suffix}.sh"
    remote_log = f"{REMOTE_STAGE}/detached_stage4_clt_{suffix}.log"
    remote_pid = f"{REMOTE_STAGE}/detached_stage4_clt_{suffix}.pid"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.asset, args.pack, args.mode, topks).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()
    cmd = f"nohup bash {remote_script} > {remote_log} 2>&1 & echo $! > {remote_pid}; cat {remote_pid}; echo {remote_log}"
    _stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        print(err)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

