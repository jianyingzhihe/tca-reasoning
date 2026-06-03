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
REMOTE_ASSETS = f"{REMOTE_STAGE}/stage2i_assets"
LOCAL_STAGE = ROOT / "doc" / "experiments" / "stage2" / "cross_model"
SELECTED_MANIFEST = LOCAL_STAGE / "stage2i_selected_12_manifest.csv"


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


def _upload_selected_manifest_and_assets(base, sftp, rows: list[dict[str, str]]) -> None:
    base._put_file(sftp, SELECTED_MANIFEST, f"{REMOTE_STAGE}/stage2i_selected_12_manifest.csv")
    print("uploaded stage2i_selected_12_manifest.csv")
    for row in rows:
        image_name = Path(row["image_filename"]).name
        image_path = Path(row["local_image_path"])
        mask_dir = Path(row["mask_dir"])
        stem = Path(image_name).stem
        if image_path.exists():
            base._put_file(sftp, image_path, f"{REMOTE_ASSETS}/images/{image_name}")
            print(f"uploaded image {image_name}")
        for mask_name in ["answer.png", "relate.png"]:
            local_mask = mask_dir / mask_name
            if local_mask.exists():
                base._put_file(sftp, local_mask, f"{REMOTE_ASSETS}/exported_masks/{stem}/{mask_name}")
                print(f"uploaded mask {stem}/{mask_name}")


def _remote_script(qwen_samples: list[str]) -> str:
    qwen_sample_arg = ",".join(qwen_samples)
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
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* | head -n 1)
LLAVA_MODEL_DIR=/root/autodl-tmp/tca-reasoning/data/modelscope_cache/swift/llava-1___5-7b-hf

echo '--- Stage 2I disk/gpu ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
.venv/bin/python -m py_compile scripts/research/run_cross_model_hidden_position_decode_smoke.py

echo '--- Stage 2I-1 Qwen selected-8 decoded bridge expansion ---'
.venv/bin/python -u scripts/research/run_cross_model_hidden_position_decode_smoke.py \\
  --model-family qwen \\
  --model-name "$QWEN_MODEL" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2i_qwen_decode_work" \\
  --sample-manifest "$STAGE/stage2i_selected_12_manifest.csv" \\
  --samples {qwen_sample_arg} \\
  --prompts B_direct,D_visual_only \\
  --layer 26 \\
  --groups top_hidden_delta_plus_answer_adjacent,answer_adjacent_text,low_delta_control,random_control_1 \\
  --directions restore \\
  --max-new-tokens 3 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 4 \\
  --out-json "$STAGE/stage2i_qwen_decoded_bridge_expansion.json" \\
  --out-csv "$STAGE/stage2i_qwen_decoded_bridge_expansion.csv"

echo '--- Stage 2I-2 LLaVA selected-12 generation-gap screen ---'
.venv/bin/python -u scripts/research/run_cross_model_hidden_position_decode_smoke.py \\
  --model-family llava \\
  --model-name "$LLAVA_MODEL_DIR" \\
  --annotation-roots "$ASSET_ROOT" \\
  --work-dir "$STAGE/stage2i_llava_gap_work" \\
  --sample-manifest "$STAGE/stage2i_selected_12_manifest.csv" \\
  --samples manifest \\
  --prompts B_direct,D_visual_only \\
  --layer 15 \\
  --groups "" \\
  --directions restore \\
  --max-new-tokens 3 \\
  --default-position-count 32 \\
  --max-evidence-positions 64 \\
  --answer-adjacent-count 4 \\
  --random-controls 4 \\
  --out-json "$STAGE/stage2i_llava_generation_gap_screen.json" \\
  --out-csv "$STAGE/stage2i_llava_generation_gap_screen.csv"
"""


def _fetch_outputs(base, sftp) -> None:
    LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
    for name in [
        "stage2i_qwen_decoded_bridge_expansion.json",
        "stage2i_qwen_decoded_bridge_expansion.csv",
        "stage2i_llava_generation_gap_screen.json",
        "stage2i_llava_generation_gap_screen.csv",
    ]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_STAGE / name))
            print(f"fetched {name}")
        except FileNotFoundError:
            print(f"missing {name}")


def _run_local_analysis() -> None:
    inputs = [
        LOCAL_STAGE / "stage2i_qwen_decoded_bridge_expansion.csv",
        LOCAL_STAGE / "stage2i_llava_generation_gap_screen.csv",
    ]
    existing = [str(path) for path in inputs if path.exists()]
    if not existing:
        print("skip analysis: no Stage 2I CSV outputs fetched")
        return
    cmd = [
        sys.executable,
        str(ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / "analyze_stage2i_cross_model_expansion.py"),
        "--inputs",
        ",".join(existing),
        "--out-summary",
        str(LOCAL_STAGE / "stage2i_cross_model_expansion_summary.csv"),
        "--out-case-table",
        str(LOCAL_STAGE / "stage2i_cross_model_expansion_case_table.csv"),
        "--out-decision",
        str(LOCAL_STAGE / "stage2i_cross_model_expansion_decision.json"),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    rows = _read_selected()
    if len(rows) < 8:
        raise RuntimeError("Stage 2I selected manifest has fewer than 8 rows.")
    qwen_samples = [row["sample_id"] for row in rows[:8]]

    base = _load_base_runner()
    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    base._upload_scripts(sftp)
    _upload_research_script(base, sftp, "run_cross_model_hidden_position_patch_smoke.py")
    _upload_research_script(base, sftp, "run_cross_model_hidden_position_decode_smoke.py")
    _upload_selected_manifest_and_assets(base, sftp, rows)
    remote_script = f"{REMOTE_STAGE}/run_stage2i_cross_model_expansion.sh"
    base._mkdir_p(sftp, REMOTE_STAGE)
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(qwen_samples).replace("\r\n", "\n"))
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
            raise TimeoutError("Stage 2I expansion exceeded 6 hours")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    while stdout.channel.recv_stderr_ready():
        print(stdout.channel.recv_stderr(8192).decode("utf-8", errors="replace"), end="")
    status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={status}")
    sftp = client.open_sftp()
    _fetch_outputs(base, sftp)
    sftp.close()
    client.close()
    if status == 0:
        _run_local_analysis()
    return status


if __name__ == "__main__":
    raise SystemExit(main())

