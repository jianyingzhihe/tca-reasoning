#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage2f_cross_model"
REMOTE_ASSETS = f"{REMOTE_STAGE}/stage2m_assets"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage2" / "cross_model"
SELECTED_MANIFEST = LOCAL_STAGE / "stage2m_selected_24_manifest.csv"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_selected() -> list[dict[str, str]]:
    with SELECTED_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}")


def _upload_assets(base, sftp, rows: list[dict[str, str]]) -> None:
    base._put_file(sftp, SELECTED_MANIFEST, f"{REMOTE_STAGE}/stage2m_selected_24_manifest.csv")
    for row in rows:
        image_name = Path(row["image_filename"]).name
        image_path = Path(row["local_image_path"])
        mask_dir = Path(row["mask_dir"])
        stem = Path(image_name).stem
        if image_path.exists():
            base._put_file(sftp, image_path, f"{REMOTE_ASSETS}/images/{image_name}")
        for mask_name in ["answer.png", "relate.png"]:
            local_mask = mask_dir / mask_name
            if local_mask.exists():
                base._put_file(sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}")
    print(f"uploaded/verified Stage 2M assets for {len(rows)} selected samples")


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
MANIFEST="$STAGE/stage2m_selected_24_manifest.csv"
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* | head -n 1)
LLAVA_MODEL_DIR=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf

echo '--- Stage 2M expanded hidden bridge disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
.venv/bin/python -m py_compile scripts/research/run_cross_model_feature_intervention_smoke.py
.venv/bin/python -m py_compile scripts/research/run_cross_model_hidden_position_patch_smoke.py
.venv/bin/python -m py_compile scripts/research/run_cross_model_wrong_target_negative_control_smoke.py
.venv/bin/python -m py_compile scripts/research/run_cross_model_mask_shuffled_negative_control_smoke.py

echo '--- Stage 2M Qwen hidden-position matched controls ---'
.venv/bin/python -u scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2m_qwen_position_work" \\
  --sample-manifest "$MANIFEST" \\
  --samples manifest \\
  --prompts B_direct,D_visual_only \\
  --layer 26 \\
  --scales 1.0 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 4 \\
  --out-json "$STAGE/stage2m_qwen_hidden_position_patch.json" \\
  --out-csv "$STAGE/stage2m_qwen_hidden_position_patch.csv"

echo '--- Stage 2M LLaVA hidden-position matched controls ---'
.venv/bin/python -u scripts/research/run_cross_model_hidden_position_patch_smoke.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2m_llava_position_work" \\
  --sample-manifest "$MANIFEST" \\
  --samples manifest \\
  --prompts B_direct,D_visual_only \\
  --layer 15 \\
  --scales 1.0 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 4 \\
  --out-json "$STAGE/stage2m_llava_hidden_position_patch.json" \\
  --out-csv "$STAGE/stage2m_llava_hidden_position_patch.csv"

echo '--- Stage 2M Qwen wrong-target negative control ---'
.venv/bin/python -u scripts/research/run_cross_model_wrong_target_negative_control_smoke.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2m_qwen_wrong_target_work" \\
  --sample-manifest "$MANIFEST" \\
  --samples manifest \\
  --prompts B_direct,D_visual_only \\
  --layer 26 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 1 \\
  --out-json "$STAGE/stage2m_qwen_wrong_target_negative_control.json" \\
  --out-csv "$STAGE/stage2m_qwen_wrong_target_negative_control.csv"

echo '--- Stage 2M LLaVA wrong-target negative control ---'
.venv/bin/python -u scripts/research/run_cross_model_wrong_target_negative_control_smoke.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2m_llava_wrong_target_work" \\
  --sample-manifest "$MANIFEST" \\
  --samples manifest \\
  --prompts B_direct,D_visual_only \\
  --layer 15 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 1 \\
  --out-json "$STAGE/stage2m_llava_wrong_target_negative_control.json" \\
  --out-csv "$STAGE/stage2m_llava_wrong_target_negative_control.csv"

echo '--- Stage 2M Qwen mask-shuffled negative control ---'
.venv/bin/python -u scripts/research/run_cross_model_mask_shuffled_negative_control_smoke.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2m_qwen_mask_shuffled_work" \\
  --sample-manifest "$MANIFEST" \\
  --samples manifest \\
  --prompts B_direct,D_visual_only \\
  --layer 26 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 1 \\
  --out-json "$STAGE/stage2m_qwen_mask_shuffled_negative_control.json" \\
  --out-csv "$STAGE/stage2m_qwen_mask_shuffled_negative_control.csv"

echo '--- Stage 2M LLaVA mask-shuffled negative control ---'
.venv/bin/python -u scripts/research/run_cross_model_mask_shuffled_negative_control_smoke.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2m_llava_mask_shuffled_work" \\
  --sample-manifest "$MANIFEST" \\
  --samples manifest \\
  --prompts B_direct,D_visual_only \\
  --layer 15 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 1 \\
  --out-json "$STAGE/stage2m_llava_mask_shuffled_negative_control.json" \\
  --out-csv "$STAGE/stage2m_llava_mask_shuffled_negative_control.csv"
"""


def _run_local_analysis() -> None:
    scripts = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2h_hidden_position_patch.py"),
            "--artifact-dir",
            str(LOCAL_STAGE),
            "--qwen-csv",
            "stage2m_qwen_hidden_position_patch.csv",
            "--llava-csv",
            "stage2m_llava_hidden_position_patch.csv",
            "--out-baseline-csv",
            str(LOCAL_STAGE / "stage2m_hidden_position_patch_baseline_gap.csv"),
            "--out-summary-csv",
            str(LOCAL_STAGE / "stage2m_hidden_position_patch_summary.csv"),
            "--out-specificity-csv",
            str(LOCAL_STAGE / "stage2m_hidden_position_patch_specificity.csv"),
            "--out-case-csv",
            str(LOCAL_STAGE / "stage2m_hidden_position_patch_case_table.csv"),
            "--out-decision-json",
            str(LOCAL_STAGE / "stage2m_hidden_position_patch_decision.json"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2k_matched_control_explanation.py"),
            "--qwen-csv",
            str(LOCAL_STAGE / "stage2m_qwen_hidden_position_patch.csv"),
            "--llava-csv",
            str(LOCAL_STAGE / "stage2m_llava_hidden_position_patch.csv"),
            "--manifest",
            str(SELECTED_MANIFEST),
            "--out-case",
            str(LOCAL_STAGE / "stage2m_matched_control_case.csv"),
            "--out-model-summary",
            str(LOCAL_STAGE / "stage2m_matched_control_model_summary.csv"),
            "--out-typed-summary",
            str(LOCAL_STAGE / "stage2m_matched_control_typed_summary.csv"),
            "--out-decision",
            str(LOCAL_STAGE / "stage2m_matched_control_decision.json"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2l_negative_controls.py"),
            "--cross-dir",
            str(LOCAL_STAGE),
            "--qwen-csv",
            "stage2m_qwen_wrong_target_negative_control.csv",
            "--llava-csv",
            "stage2m_llava_wrong_target_negative_control.csv",
            "--out-case-csv",
            str(LOCAL_STAGE / "stage2m_wrong_target_case_table.csv"),
            "--out-summary-csv",
            str(LOCAL_STAGE / "stage2m_wrong_target_summary.csv"),
            "--out-decision-json",
            str(LOCAL_STAGE / "stage2m_wrong_target_decision.json"),
            "--out-doc",
            str(ROOT / "doc" / "experiments" / "stage2" / "stage2m_wrong_target_tmp.md"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2l_mask_shuffled_control.py"),
            "--cross-dir",
            str(LOCAL_STAGE),
            "--qwen-csv",
            "stage2m_qwen_mask_shuffled_negative_control.csv",
            "--llava-csv",
            "stage2m_llava_mask_shuffled_negative_control.csv",
            "--out-case-csv",
            str(LOCAL_STAGE / "stage2m_mask_shuffled_case_table.csv"),
            "--out-summary-csv",
            str(LOCAL_STAGE / "stage2m_mask_shuffled_summary.csv"),
            "--out-decision-json",
            str(LOCAL_STAGE / "stage2m_mask_shuffled_decision.json"),
            "--out-doc",
            str(ROOT / "doc" / "experiments" / "stage2" / "stage2m_mask_shuffled_tmp.md"),
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        [sys.executable, str(scripts / "analyze_stage2m_full_replication.py")],
        cwd=str(ROOT),
        check=True,
    )


def main() -> int:
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    rows = _read_selected()
    base = _load_base_runner()
    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    base._upload_scripts(sftp)
    for name in [
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_cross_model_wrong_target_negative_control_smoke.py",
        "run_cross_model_mask_shuffled_negative_control_smoke.py",
    ]:
        _upload_research_script(base, sftp, name)
    _upload_assets(base, sftp, rows)
    remote_script = f"{REMOTE_STAGE}/run_stage2m_hidden_bridge.sh"
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
        if time.time() - start > 43200:
            stdout.channel.close()
            raise TimeoutError("Stage 2M hidden bridge exceeded 12 hours")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={status}")

    sftp = client.open_sftp()
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    fetch_names = [
        "stage2m_qwen_hidden_position_patch.json",
        "stage2m_qwen_hidden_position_patch.csv",
        "stage2m_llava_hidden_position_patch.json",
        "stage2m_llava_hidden_position_patch.csv",
        "stage2m_qwen_wrong_target_negative_control.json",
        "stage2m_qwen_wrong_target_negative_control.csv",
        "stage2m_llava_wrong_target_negative_control.json",
        "stage2m_llava_wrong_target_negative_control.csv",
        "stage2m_qwen_mask_shuffled_negative_control.json",
        "stage2m_qwen_mask_shuffled_negative_control.csv",
        "stage2m_llava_mask_shuffled_negative_control.json",
        "stage2m_llava_mask_shuffled_negative_control.csv",
    ]
    for name in fetch_names:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_STAGE / name))
            print(f"fetched {name}")
        except FileNotFoundError:
            print(f"missing {name}")
    sftp.close()
    client.close()
    if status == 0:
        _run_local_analysis()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
