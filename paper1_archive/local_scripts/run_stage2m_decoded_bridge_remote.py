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
REMOTE_ASSETS = f"{REMOTE_STAGE}/stage2m_decoded_assets"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage2" / "cross_model"
SELECTED_MANIFEST = LOCAL_STAGE / "stage2m_decoded_candidate_manifest.csv"
PROMPT_ROWS = LOCAL_STAGE / "stage2m_decoded_candidate_prompt_rows.csv"


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


def _model_samples(model: str) -> list[str]:
    rows = _read_csv(PROMPT_ROWS)
    selected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("model_family") != model:
            continue
        sample_id = row.get("sample_id", "")
        if sample_id and sample_id not in seen:
            selected.append(sample_id)
            seen.add(sample_id)
    return selected


def _upload_research_script(base, sftp, name: str) -> None:
    local = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
    remote = f"{REMOTE_ROOT}/scripts/research/{name}"
    base._put_file(sftp, local, remote)
    sftp.chmod(remote, 0o755)
    print(f"uploaded {name}")


def _upload_assets(base, sftp, rows: list[dict[str, str]]) -> None:
    base._put_file(sftp, SELECTED_MANIFEST, f"{REMOTE_STAGE}/stage2m_decoded_candidate_manifest.csv")
    base._put_file(sftp, PROMPT_ROWS, f"{REMOTE_STAGE}/stage2m_decoded_candidate_prompt_rows.csv")
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
    print(f"uploaded/verified Stage 2M decoded assets for {len(rows)} selected samples")


def _remote_script(qwen_samples: list[str], llava_samples: list[str]) -> str:
    qwen_csv = ",".join(qwen_samples)
    llava_csv = ",".join(llava_samples)
    groups = (
        "top_hidden_delta_plus_answer_adjacent,"
        "delta_matched_plus_answer_adjacent,"
        "activation_matched_plus_answer_adjacent,"
        "answer_adjacent_text,"
        "low_delta_control,"
        "random_control_1,"
        "evidence_region_plus_answer_adjacent"
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
MANIFEST="$STAGE/stage2m_decoded_candidate_manifest.csv"
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* | head -n 1)
LLAVA_MODEL_DIR=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf

echo '--- Stage 2M decoded bridge disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
.venv/bin/python -m py_compile scripts/research/run_cross_model_feature_intervention_smoke.py
.venv/bin/python -m py_compile scripts/research/run_cross_model_hidden_position_patch_smoke.py
.venv/bin/python -m py_compile scripts/research/run_cross_model_hidden_position_decode_smoke.py

echo '--- Stage 2M Qwen decoded bridge on passing cases ---'
.venv/bin/python -u scripts/research/run_cross_model_hidden_position_decode_smoke.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2m_qwen_decoded_work" \\
  --sample-manifest "$MANIFEST" \\
  --samples "{qwen_csv}" \\
  --prompts B_direct,D_visual_only \\
  --layer 26 \\
  --groups "{groups}" \\
  --directions restore \\
  --max-new-tokens 3 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 4 \\
  --out-json "$STAGE/stage2m_qwen_decoded_bridge.json" \\
  --out-csv "$STAGE/stage2m_qwen_decoded_bridge.csv"

echo '--- Stage 2M LLaVA decoded bridge on passing cases ---'
.venv/bin/python -u scripts/research/run_cross_model_hidden_position_decode_smoke.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2m_llava_decoded_work" \\
  --sample-manifest "$MANIFEST" \\
  --samples "{llava_csv}" \\
  --prompts B_direct,D_visual_only \\
  --layer 15 \\
  --groups "{groups}" \\
  --directions restore \\
  --max-new-tokens 3 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 4 \\
  --out-json "$STAGE/stage2m_llava_decoded_bridge.json" \\
  --out-csv "$STAGE/stage2m_llava_decoded_bridge.csv"
"""


def _run_local_analysis() -> None:
    scripts = ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze_stage2h_decoded_answer_smoke.py"),
            "--inputs",
            ",".join(
                [
                    str(LOCAL_STAGE / "stage2m_qwen_decoded_bridge.csv"),
                    str(LOCAL_STAGE / "stage2m_llava_decoded_bridge.csv"),
                ]
            ),
            "--out-summary",
            str(LOCAL_STAGE / "stage2m_decoded_bridge_summary.csv"),
            "--out-case-table",
            str(LOCAL_STAGE / "stage2m_decoded_bridge_case_table.csv"),
            "--out-decision",
            str(LOCAL_STAGE / "stage2m_decoded_bridge_decision.json"),
        ],
        cwd=str(ROOT),
        check=True,
    )


def main() -> int:
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    if not SELECTED_MANIFEST.exists() or not PROMPT_ROWS.exists():
        raise FileNotFoundError("Run scripts/local/build_stage2m_decoded_candidate_manifest.py first.")
    rows = _read_csv(SELECTED_MANIFEST)
    qwen_samples = _model_samples("qwen")
    llava_samples = _model_samples("llava")
    if not qwen_samples or not llava_samples:
        raise RuntimeError("Decoded candidate list must contain both qwen and llava samples.")

    base = _load_base_runner()
    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    base._upload_scripts(sftp)
    for name in [
        "run_cross_model_feature_intervention_smoke.py",
        "run_cross_model_hidden_position_patch_smoke.py",
        "run_cross_model_hidden_position_decode_smoke.py",
    ]:
        _upload_research_script(base, sftp, name)
    _upload_assets(base, sftp, rows)
    remote_script = f"{REMOTE_STAGE}/run_stage2m_decoded_bridge.sh"
    base._mkdir_p(sftp, REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(qwen_samples, llava_samples).replace("\r\n", "\n"))
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
            raise TimeoutError("Stage 2M decoded bridge exceeded 6 hours")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={status}")

    sftp = client.open_sftp()
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    for name in [
        "stage2m_qwen_decoded_bridge.json",
        "stage2m_qwen_decoded_bridge.csv",
        "stage2m_llava_decoded_bridge.json",
        "stage2m_llava_decoded_bridge.csv",
    ]:
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
