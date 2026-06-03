#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage2f_cross_model"
REMOTE_ASSETS = f"{REMOTE_STAGE}/stage2o_assets"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage2" / "cross_model"
ALL52_MANIFEST = LOCAL_STAGE / "stage2n_all52_manifest.csv"
SPECIFICITY = LOCAL_STAGE / "stage2n_hidden_specificity_case.csv"
QWEN_RUN_MANIFEST = LOCAL_STAGE / "stage2o_qwen_feature_route_run_manifest.csv"
LLAVA_RUN_MANIFEST = LOCAL_STAGE / "stage2o_llava_feature_route_run_manifest.csv"
SAMPLE_MANIFEST = LOCAL_STAGE / "stage2o_feature_route_sample_manifest.csv"


FETCH_FILES = [
    "stage2o_qwen_attribution_feature_bridge.csv",
    "stage2o_qwen_attribution_feature_bridge.json",
    "stage2o_llava_attribution_feature_bridge.csv",
    "stage2o_llava_attribution_feature_bridge.json",
    "stage2o_qwen_source_control_probe.csv",
    "stage2o_qwen_source_control_probe.json",
    "stage2o_llava_source_control_probe.csv",
    "stage2o_llava_source_control_probe.json",
]


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _select_runs(model: str, count: int) -> list[dict[str, str]]:
    rows = [
        row
        for row in _read_csv(SPECIFICITY)
        if row.get("model_family") == model
        and row.get("mask_condition") == "union_mask"
        and row.get("direction") == "restore"
    ]
    rows.sort(key=lambda row: float(row.get("source_minus_random_logit", "0") or 0.0), reverse=True)
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["sample_id"], row["prompt_name"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "sample_id": row["sample_id"],
                "prompt_name": row["prompt_name"],
                "model_family": model,
                "stage2n_source_minus_random_logit": row.get("source_minus_random_logit", ""),
                "stage2n_source_effect_logit": row.get("source_effect_logit", ""),
            }
        )
        if len(selected) >= count:
            break
    return selected


def _prepare_manifests() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    qwen = _select_runs("qwen", 12)
    llava = _select_runs("llava", 8)
    all52 = _read_csv(ALL52_MANIFEST)
    needed_ids = {row["sample_id"] for row in qwen + llava}
    sample_rows = [row for row in all52 if row.get("sample_id") in needed_ids]
    _write_csv(
        QWEN_RUN_MANIFEST,
        qwen,
        ["sample_id", "prompt_name", "model_family", "stage2n_source_minus_random_logit", "stage2n_source_effect_logit"],
    )
    _write_csv(
        LLAVA_RUN_MANIFEST,
        llava,
        ["sample_id", "prompt_name", "model_family", "stage2n_source_minus_random_logit", "stage2n_source_effect_logit"],
    )
    _write_csv(SAMPLE_MANIFEST, sample_rows, list(all52[0].keys()))
    (LOCAL_STAGE / "stage2o_feature_route_selection_summary.json").write_text(
        json.dumps(
            {
                "qwen_run_count": len(qwen),
                "llava_run_count": len(llava),
                "sample_count": len(sample_rows),
                "sample_ids": sorted(needed_ids),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return qwen, llava, sample_rows


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}")


def _upload_assets(base, sftp, sample_rows: list[dict[str, str]]) -> None:
    base._put_file(sftp, SAMPLE_MANIFEST, f"{REMOTE_STAGE}/stage2o_feature_route_sample_manifest.csv")
    base._put_file(sftp, QWEN_RUN_MANIFEST, f"{REMOTE_STAGE}/stage2o_qwen_feature_route_run_manifest.csv")
    base._put_file(sftp, LLAVA_RUN_MANIFEST, f"{REMOTE_STAGE}/stage2o_llava_feature_route_run_manifest.csv")
    for row in sample_rows:
        image_path = Path(row["local_image_path"])
        mask_dir = Path(row["mask_dir"])
        image_name = Path(row["image_filename"]).name
        stem = Path(image_name).stem
        if image_path.exists():
            base._put_file(sftp, image_path, f"{REMOTE_ASSETS}/images/{image_name}")
        for mask_name in ["answer.png", "relate.png"]:
            local_mask = mask_dir / mask_name
            if local_mask.exists():
                base._put_file(sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}")
    print(f"uploaded Stage 2O assets for {len(sample_rows)} samples")


def _remote_script() -> str:
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
SAMPLE_MANIFEST="$STAGE/stage2o_feature_route_sample_manifest.csv"
QWEN_RUNS="$STAGE/stage2o_qwen_feature_route_run_manifest.csv"
LLAVA_RUNS="$STAGE/stage2o_llava_feature_route_run_manifest.csv"
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* | head -n 1)
LLAVA_MODEL_DIR=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf

echo '--- Stage 2O disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
.venv/bin/python -m py_compile \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py

echo '--- Stage 2O-1 Qwen attribution-weighted feature bridge ---'
.venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2o_qwen_feature_work" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$QWEN_RUNS" \\
  --layer 26 \\
  --mask-condition union_mask \\
  --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage2o_qwen_attribution_feature_bridge.json" \\
  --out-csv "$STAGE/stage2o_qwen_attribution_feature_bridge.csv"

echo '--- Stage 2O-1 LLaVA attribution-weighted feature bridge ---'
.venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --transcoder-ref KokosDev/llava15-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2o_llava_feature_work" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$LLAVA_RUNS" \\
  --layer 15 \\
  --mask-condition union_mask \\
  --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage2o_llava_attribution_feature_bridge.json" \\
  --out-csv "$STAGE/stage2o_llava_attribution_feature_bridge.csv"

echo '--- Stage 2O-2 Qwen approximate source-control route probe ---'
.venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2o_qwen_source_work" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$QWEN_RUNS" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask \\
  --position-group top_hidden_delta_plus_answer_adjacent \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage2o_qwen_source_control_probe.json" \\
  --out-csv "$STAGE/stage2o_qwen_source_control_probe.csv"

echo '--- Stage 2O-2 LLaVA approximate source-control route probe ---'
.venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --transcoder-ref KokosDev/llava15-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2o_llava_source_work" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$LLAVA_RUNS" \\
  --layer 15 \\
  --mask-conditions answer_mask,union_mask \\
  --position-group top_hidden_delta_plus_answer_adjacent \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage2o_llava_source_control_probe.json" \\
  --out-csv "$STAGE/stage2o_llava_source_control_probe.csv"
"""


def _fetch_outputs(sftp) -> None:
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    for name in FETCH_FILES:
        remote = f"{REMOTE_STAGE}/{name}"
        local = LOCAL_STAGE / name
        try:
            sftp.get(remote, str(local))
            print(f"fetched {name}")
        except FileNotFoundError:
            print(f"missing {name}")


def _run_local_analysis() -> None:
    scripts = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2o_feature_bridge.py"),
            "--inputs",
            ",".join(
                [
                    str(LOCAL_STAGE / "stage2o_qwen_attribution_feature_bridge.csv"),
                    str(LOCAL_STAGE / "stage2o_llava_attribution_feature_bridge.csv"),
                ]
            ),
            "--out-summary",
            str(LOCAL_STAGE / "stage2o_feature_bridge_summary.csv"),
            "--out-specificity",
            str(LOCAL_STAGE / "stage2o_feature_bridge_specificity.csv"),
            "--out-decision",
            str(LOCAL_STAGE / "stage2o_feature_bridge_decision.json"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2o_source_control_probe.py"),
            "--inputs",
            ",".join(
                [
                    str(LOCAL_STAGE / "stage2o_qwen_source_control_probe.csv"),
                    str(LOCAL_STAGE / "stage2o_llava_source_control_probe.csv"),
                ]
            ),
            "--out-summary",
            str(LOCAL_STAGE / "stage2o_source_control_summary.csv"),
            "--out-specificity",
            str(LOCAL_STAGE / "stage2o_source_control_specificity.csv"),
            "--out-mask-specificity",
            str(LOCAL_STAGE / "stage2o_source_control_mask_specificity.csv"),
            "--out-decision",
            str(LOCAL_STAGE / "stage2o_source_control_decision.json"),
        ],
        cwd=str(ROOT),
        check=True,
    )


def main() -> int:
    base = _load_base_runner()
    qwen, llava, sample_rows = _prepare_manifests()
    print(f"selected qwen_runs={len(qwen)} llava_runs={len(llava)} samples={len(sample_rows)}")

    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username="root",
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    sftp = client.open_sftp()
    for name in [
        "run_stage2o_attribution_weighted_feature_bridge.py",
        "run_stage2o_cross_model_source_control_probe.py",
        "analyze_stage2o_feature_bridge.py",
        "analyze_stage2o_source_control_probe.py",
    ]:
        _upload_research_script(base, sftp, name)
    _upload_assets(base, sftp, sample_rows)
    remote_script = f"{REMOTE_STAGE}/run_stage2o_feature_route.sh"
    base._mkdir_p(sftp, REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script().replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
    stdout.channel.settimeout(0.0)
    stderr.channel.settimeout(0.0)
    start = time.time()
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
        if time.time() - start > 21600:
            stdout.channel.close()
            raise TimeoutError("remote Stage 2O command exceeded 6 hours")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")
    sftp = client.open_sftp()
    _fetch_outputs(sftp)
    sftp.close()
    client.close()
    if exit_status == 0:
        _run_local_analysis()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
