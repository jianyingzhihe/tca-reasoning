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
REMOTE_ASSETS = f"{REMOTE_STAGE}/stage2p_assets"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage2" / "cross_model"
ALL52_MANIFEST = LOCAL_STAGE / "stage2n_all52_manifest.csv"
SPECIFICITY = LOCAL_STAGE / "stage2n_hidden_specificity_case.csv"
STAGE2O_SELECTION = LOCAL_STAGE / "stage2o_feature_route_selection_summary.json"
QWEN_RUN_MANIFEST = LOCAL_STAGE / "stage2p_qwen_heldout_run_manifest.csv"
LLAVA_RUN_MANIFEST = LOCAL_STAGE / "stage2p_llava_diagnostic_run_manifest.csv"
SAMPLE_MANIFEST = LOCAL_STAGE / "stage2p_cross_modal_sample_manifest.csv"

LLAVA_LAYERS = [12, 15, 18, 21]
LLAVA_TOPKS = [1, 8, 32]


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


def _stage2o_sample_ids() -> set[str]:
    if not STAGE2O_SELECTION.exists():
        return set()
    data = json.loads(STAGE2O_SELECTION.read_text(encoding="utf-8"))
    return set(data.get("sample_ids", []))


def _select_qwen_heldout(count: int = 24) -> list[dict[str, str]]:
    excluded_samples = _stage2o_sample_ids()
    rows = [
        row
        for row in _read_csv(SPECIFICITY)
        if row.get("model_family") == "qwen"
        and row.get("mask_condition") == "union_mask"
        and row.get("direction") == "restore"
        and row.get("sample_id") not in excluded_samples
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
                "model_family": "qwen",
                "stage2n_source_minus_random_logit": row.get("source_minus_random_logit", ""),
                "stage2n_source_effect_logit": row.get("source_effect_logit", ""),
                "stage2p_selection_note": "heldout_excludes_stage2o_samples",
            }
        )
        if len(selected) >= count:
            break
    return selected


def _select_llava_diagnostic() -> list[dict[str, str]]:
    path = LOCAL_STAGE / "stage2o_llava_feature_route_run_manifest.csv"
    rows = _read_csv(path)
    return [
        {
            "sample_id": row["sample_id"],
            "prompt_name": row["prompt_name"],
            "model_family": "llava",
            "stage2n_source_minus_random_logit": row.get("stage2n_source_minus_random_logit", ""),
            "stage2n_source_effect_logit": row.get("stage2n_source_effect_logit", ""),
            "stage2p_selection_note": "stage2o_llava_diagnostic_reuse",
        }
        for row in rows
    ]


def _prepare_manifests() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    qwen = _select_qwen_heldout(24)
    llava = _select_llava_diagnostic()
    all52 = _read_csv(ALL52_MANIFEST)
    needed_ids = {row["sample_id"] for row in qwen + llava}
    sample_rows = [row for row in all52 if row.get("sample_id") in needed_ids]
    fields = [
        "sample_id",
        "prompt_name",
        "model_family",
        "stage2n_source_minus_random_logit",
        "stage2n_source_effect_logit",
        "stage2p_selection_note",
    ]
    _write_csv(QWEN_RUN_MANIFEST, qwen, fields)
    _write_csv(LLAVA_RUN_MANIFEST, llava, fields)
    _write_csv(SAMPLE_MANIFEST, sample_rows, list(all52[0].keys()))
    (LOCAL_STAGE / "stage2p_cross_modal_selection_summary.json").write_text(
        json.dumps(
            {
                "qwen_heldout_run_count": len(qwen),
                "llava_diagnostic_run_count": len(llava),
                "sample_count": len(sample_rows),
                "excluded_stage2o_sample_ids": sorted(_stage2o_sample_ids()),
                "selected_sample_ids": sorted(needed_ids),
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
    print(f"uploaded {name}", flush=True)


def _upload_assets(base, sftp, sample_rows: list[dict[str, str]]) -> None:
    base._put_file(sftp, SAMPLE_MANIFEST, f"{REMOTE_STAGE}/stage2p_cross_modal_sample_manifest.csv")
    base._put_file(sftp, QWEN_RUN_MANIFEST, f"{REMOTE_STAGE}/stage2p_qwen_heldout_run_manifest.csv")
    base._put_file(sftp, LLAVA_RUN_MANIFEST, f"{REMOTE_STAGE}/stage2p_llava_diagnostic_run_manifest.csv")
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
    print(f"uploaded Stage 2P assets for {len(sample_rows)} samples", flush=True)


def _llava_sweep_commands() -> str:
    chunks: list[str] = []
    for layer in LLAVA_LAYERS:
        for topk in LLAVA_TOPKS:
            stem = f"stage2p_llava_layer{layer}_top{topk}_feature_bridge"
            chunks.append(
                f"""
echo '--- Stage 2P-2 LLaVA layer {layer} top{topk} feature diagnostic ---'
set +e
.venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --transcoder-ref KokosDev/llava15-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2p_llava_l{layer}_top{topk}_work" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$LLAVA_RUNS" \\
  --layer {layer} \\
  --mask-condition union_mask \\
  --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
  --top-k-features {topk} \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/{stem}.json" \\
  --out-csv "$STAGE/{stem}.csv"
STATUS=$?
set -e
echo "stage2p_llava_layer{layer}_top{topk}_exit=$STATUS"
"""
            )
    return "\n".join(chunks)


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
SAMPLE_MANIFEST="$STAGE/stage2p_cross_modal_sample_manifest.csv"
QWEN_RUNS="$STAGE/stage2p_qwen_heldout_run_manifest.csv"
LLAVA_RUNS="$STAGE/stage2p_llava_diagnostic_run_manifest.csv"
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* | head -n 1)
LLAVA_MODEL_DIR=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf

echo '--- Stage 2P disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
.venv/bin/python -m py_compile \\
  scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  scripts/research/run_stage2o_cross_model_source_control_probe.py

echo '--- Stage 2P-1 Qwen heldout attribution feature bridge: answer_mask ---'
.venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2p_qwen_feature_answer_work" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$QWEN_RUNS" \\
  --layer 26 \\
  --mask-condition answer_mask \\
  --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage2p_qwen_feature_answer.json" \\
  --out-csv "$STAGE/stage2p_qwen_feature_answer.csv"

echo '--- Stage 2P-1 Qwen heldout attribution feature bridge: union_mask ---'
.venv/bin/python -u scripts/research/run_stage2o_attribution_weighted_feature_bridge.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2p_qwen_feature_union_work" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$QWEN_RUNS" \\
  --layer 26 \\
  --mask-condition union_mask \\
  --position-groups top_hidden_delta_plus_answer_adjacent,top_hidden_delta,answer_adjacent_text \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage2p_qwen_feature_union.json" \\
  --out-csv "$STAGE/stage2p_qwen_feature_union.csv"

echo '--- Stage 2P-1 Qwen heldout approximate source-control route probe ---'
.venv/bin/python -u scripts/research/run_stage2o_cross_model_source_control_probe.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --transcoder-ref KokosDev/qwen2p5vl-7b-clt \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2p_qwen_source_work" \\
  --sample-manifest "$SAMPLE_MANIFEST" \\
  --run-manifest "$QWEN_RUNS" \\
  --layer 26 \\
  --mask-conditions answer_mask,union_mask \\
  --position-group top_hidden_delta_plus_answer_adjacent \\
  --top-k-features 8 \\
  --control-pool-size 2048 \\
  --out-json "$STAGE/stage2p_qwen_source_control_probe.json" \\
  --out-csv "$STAGE/stage2p_qwen_source_control_probe.csv"

echo '--- Stage 2P LLaVA CLT predownload/check after Qwen primary path ---'
for LAYER in {' '.join(str(x) for x in LLAVA_LAYERS)}; do
  echo "llava_clt_layer=$LAYER download_attempt_with_network_turbo"
  set +e
  timeout 1200 .venv/bin/python - "$LAYER" <<'PY'
import sys
from huggingface_hub import hf_hub_download
layer = int(sys.argv[1])
name = f"transcoder_L{{layer}}.pt"
try:
    path = hf_hub_download(repo_id="KokosDev/llava15-7b-clt", filename=name, local_files_only=False)
    print(f"llava_clt_layer={{layer}} status=ok path={{path}}", flush=True)
except Exception as exc:
    print(f"llava_clt_layer={{layer}} status=blocked error={{type(exc).__name__}}: {{exc}}", flush=True)
    raise
PY
  STATUS=$?
  set -e
  echo "llava_clt_layer=$LAYER download_exit=$STATUS"
done

{_llava_sweep_commands()}
"""


def _fetch_outputs(sftp) -> None:
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    files = [
        "stage2p_qwen_feature_answer.csv",
        "stage2p_qwen_feature_answer.json",
        "stage2p_qwen_feature_union.csv",
        "stage2p_qwen_feature_union.json",
        "stage2p_qwen_source_control_probe.csv",
        "stage2p_qwen_source_control_probe.json",
    ]
    for layer in LLAVA_LAYERS:
        for topk in LLAVA_TOPKS:
            files.extend(
                [
                    f"stage2p_llava_layer{layer}_top{topk}_feature_bridge.csv",
                    f"stage2p_llava_layer{layer}_top{topk}_feature_bridge.json",
                ]
            )
    for name in files:
        remote = f"{REMOTE_STAGE}/{name}"
        local = LOCAL_STAGE / name
        try:
            sftp.get(remote, str(local))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def _existing_llava_csvs() -> list[str]:
    files: list[str] = []
    for layer in LLAVA_LAYERS:
        for topk in LLAVA_TOPKS:
            path = LOCAL_STAGE / f"stage2p_llava_layer{layer}_top{topk}_feature_bridge.csv"
            if path.exists():
                files.append(str(path))
    return files


def _run_local_analysis() -> None:
    scripts = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2o_feature_bridge.py"),
            "--inputs",
            ",".join(
                [
                    str(LOCAL_STAGE / "stage2p_qwen_feature_answer.csv"),
                    str(LOCAL_STAGE / "stage2p_qwen_feature_union.csv"),
                ]
            ),
            "--out-summary",
            str(LOCAL_STAGE / "stage2p_qwen_feature_bridge_summary.csv"),
            "--out-specificity",
            str(LOCAL_STAGE / "stage2p_qwen_feature_bridge_specificity.csv"),
            "--out-decision",
            str(LOCAL_STAGE / "stage2p_qwen_feature_bridge_decision.json"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2o_source_control_probe.py"),
            "--inputs",
            str(LOCAL_STAGE / "stage2p_qwen_source_control_probe.csv"),
            "--out-summary",
            str(LOCAL_STAGE / "stage2p_qwen_source_control_summary.csv"),
            "--out-specificity",
            str(LOCAL_STAGE / "stage2p_qwen_source_control_specificity.csv"),
            "--out-mask-specificity",
            str(LOCAL_STAGE / "stage2p_qwen_source_control_mask_specificity.csv"),
            "--out-decision",
            str(LOCAL_STAGE / "stage2p_qwen_source_control_decision.json"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    llava_csvs = _existing_llava_csvs()
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2p_llava_layer_sweep.py"),
            "--inputs",
            ",".join(llava_csvs),
            "--expected-layers",
            ",".join(str(x) for x in LLAVA_LAYERS),
            "--out-summary",
            str(LOCAL_STAGE / "stage2p_llava_layer_sweep_summary.csv"),
            "--out-specificity",
            str(LOCAL_STAGE / "stage2p_llava_layer_sweep_specificity.csv"),
            "--out-decision",
            str(LOCAL_STAGE / "stage2p_llava_layer_sweep_decision.json"),
        ],
        cwd=str(ROOT),
        check=True,
    )


def main() -> int:
    base = _load_base_runner()
    qwen, llava, sample_rows = _prepare_manifests()
    print(f"selected qwen_heldout_runs={len(qwen)} llava_diag_runs={len(llava)} samples={len(sample_rows)}", flush=True)

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
        "analyze_stage2p_llava_layer_sweep.py",
    ]:
        _upload_research_script(base, sftp, name)
    _upload_assets(base, sftp, sample_rows)
    remote_script = f"{REMOTE_STAGE}/run_stage2p_cross_modal_feature.sh"
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
        if time.time() - start > 28800:
            stdout.channel.close()
            raise TimeoutError("remote Stage 2P command exceeded 8 hours")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}", flush=True)
    sftp = client.open_sftp()
    _fetch_outputs(sftp)
    sftp.close()
    client.close()
    if exit_status == 0:
        _run_local_analysis()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
