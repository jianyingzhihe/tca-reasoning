#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\Bridging")
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/stage6_unified_prompt_text"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_CROSS = ROOT / "doc" / "experiments" / "stage6" / "cross_model"
PREFIX = "stage6_unified_fixednode"


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


def _csv_text(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buf.getvalue()


def _manifest_for(tag: str) -> Path:
    return LOCAL_CROSS / f"{PREFIX}_{tag}_manifest.csv"


def _filter_rows(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    key = "stage6_include_smoke" if mode == "smoke" else "stage6_include_full"
    return [row for row in rows if row.get(key, "1") == "1"]


def _upload_manifest_and_assets(base, sftp, manifest: Path, mode: str, tag: str) -> str:
    rows = _filter_rows(_read_csv(manifest), mode)
    if not rows:
        raise ValueError(f"empty Gemma fixed-node manifest for mode={mode}: {manifest}")
    base._mkdir_p(sftp, REMOTE_STAGE)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    uploaded = 0
    skipped = 0
    remote_rows: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        image = Path(item.get("local_image_path", ""))
        image_name = Path(item.get("image_filename", "")).name or image.name
        if not image.exists():
            raise FileNotFoundError(image)
        remote_image = f"{REMOTE_ASSETS}/images/{image_name}"
        try:
            sftp.stat(remote_image)
            skipped += 1
        except OSError:
            base._put_file(sftp, image, remote_image)
            uploaded += 1
        item["remote_image_path"] = remote_image
        remote_rows.append(item)
    remote_manifest = f"{REMOTE_STAGE}/{PREFIX}_{mode}_{tag}_manifest.csv"
    with sftp.file(remote_manifest, "w") as handle:
        handle.write(_csv_text(remote_rows, list(remote_rows[0].keys())).replace("\r\n", "\n"))
    print(f"uploaded Gemma fixed-node manifest rows={len(remote_rows)} images_uploaded={uploaded} skipped={skipped}", flush=True)
    return remote_manifest


def _remote_script(mode: str, tag: str, remote_manifest: str, aligned_only: bool, max_rows: int) -> str:
    aligned_filter = "1" if aligned_only else "0"
    return f"""#!/usr/bin/env bash
set -e
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub
STAGE={REMOTE_STAGE}
export OUT="$STAGE/{PREFIX}_{mode}_{tag}_raw.csv"
export SUMMARY="$STAGE/{PREFIX}_{mode}_{tag}_run.json"
mkdir -p "$STAGE"

echo '--- Stage6 unified Gemma fixed-node preflight ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
echo '--- Stage6 unified Gemma fixed-node cgroup memory before cache evict ---'
cat /sys/fs/cgroup/memory.max 2>/dev/null || true
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.stat 2>/dev/null | egrep '^(anon|file|active_file|inactive_file|slab)' || true

.venv/bin/python - <<'PY'
import os
import time

roots = [
    "/root/autodl-tmp/tca-reasoning/data/hf_cache",
    "/root/autodl-tmp/tca-reasoning/data/modelscope_cache",
    "/root/autodl-tmp/tca-reasoning/stage6_unified_prompt_text",
]
has_fadvise = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
count = 0
bytes_seen = 0
ok = 0
err = 0
for root in roots:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                size = os.stat(path).st_size
            except OSError:
                continue
            if size < 1024 * 1024:
                continue
            count += 1
            bytes_seen += size
            if not has_fadvise:
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    ok += 1
                finally:
                    os.close(fd)
            except OSError:
                err += 1
print(f"[cache-evict] posix_fadvise={{has_fadvise}} files={{count}} bytes_seen={{bytes_seen}} ok={{ok}} err={{err}}", flush=True)
time.sleep(1)
PY

echo '--- Stage6 unified Gemma fixed-node cgroup memory after cache evict ---'
cat /sys/fs/cgroup/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
cat /sys/fs/cgroup/memory.stat 2>/dev/null | egrep '^(anon|file|active_file|inactive_file|slab)' || true

.venv/bin/python - <<'PY'
import csv, json, math, os, sys, time
from pathlib import Path

import torch
from PIL import Image

from circuit_tracer import ReplacementModel
from circuit_tracer.attribution.attribute import _build_multimodal_batch
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub


manifest = Path({remote_manifest!r})
out_csv = Path(os.environ["OUT"])
summary_json = Path(os.environ["SUMMARY"])
aligned_only = bool(int({aligned_filter!r}))
max_rows = int({max_rows})


def infer_model_name(repo_id):
    from huggingface_hub import hf_hub_download
    import yaml
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {{}}
    model_name = cfg.get("model_name", "")
    if not model_name:
        raise ValueError(f"model_name missing in {{repo_id}}/config.yaml")
    return model_name


def f(raw, default=0.0):
    try:
        return float(raw) if raw not in (None, "") else default
    except Exception:
        return default


def as_int(raw, default=-1):
    try:
        return int(float(raw))
    except Exception:
        return default


def parse_controls(raw):
    out = []
    for item in (raw or "").split("|"):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            continue
        out.append((as_int(parts[0]), as_int(parts[1]), as_int(parts[2])))
    return [x for x in out if min(x) >= 0]


def to_device(model, batch):
    return {{k: (v.to(model.cfg.device) if torch.is_tensor(v) else v) for k, v in batch.items()}}


def score_token(logits, token_id):
    last = logits.shape[1] - 1
    logit = float(logits[0, last, token_id].item())
    rank = int((logits[0, last] > logits[0, last, token_id]).sum().item() + 1)
    return logit, rank


def top_wrong(logits, target_id):
    last = logits.shape[1] - 1
    scores = logits[0, last].detach().float().cpu()
    scores[target_id] = -float("inf")
    return int(torch.argmax(scores).item())


with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if max_rows > 0:
    rows = rows[:max_rows]

transcoder_set = "tianhux2/gemma3-4b-it-plt"
dtype = torch.bfloat16
transcoders, config = load_transcoder_from_hub(transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
model_name = config.get("model_name") or infer_model_name(transcoder_set)
print(f"[init] loading model={{model_name}} rows={{len(rows)}} aligned_only={{aligned_only}}")
model = ReplacementModel.from_pretrained_and_transcoders(model_name, transcoders, dtype=dtype)

fieldnames = [
    "candidate_id", "sample_id", "stage6_original_sample_id", "stage6_question_variant", "stage6_prompt_family",
    "stage6_sample_type", "source_layer", "source_pos", "source_feature_id", "source_path_mass_best",
    "target_token_id", "wrong_token_id", "target_token_same", "prefix_ok", "format_ok", "aligned_main",
    "source_damage_target_logit", "source_damage_wrong_logit", "source_minus_controls", "correct_minus_wrong",
    "control_damage_target_mean", "control_count", "before_target_logit", "after_target_logit",
    "before_target_rank", "after_target_rank", "status", "error_message",
]
results = []
ok = skipped = errors = 0
t0 = time.time()

for idx, row in enumerate(rows, start=1):
    aligned = row.get("prefix_ok") == "1" and row.get("target_token_same") == "1"
    base = {{key: row.get(key, "") for key in fieldnames}}
    base["aligned_main"] = "1" if aligned else "0"
    if aligned_only and not aligned:
        base["status"] = "skipped_unaligned"
        results.append(base)
        skipped += 1
        continue
    try:
        target_id = as_int(row.get("target_token_id"))
        layer = as_int(row.get("source_layer"))
        pos = as_int(row.get("source_pos"))
        feature = as_int(row.get("source_feature_id"))
        if min(target_id, layer, pos, feature) < 0:
            raise ValueError("missing target or feature coordinates")
        image = Image.open(row["remote_image_path"]).convert("RGB")
        batch = _build_multimodal_batch(
            model.processor,
            image,
            f"<start_of_image> {{row['question']}}",
            assistant_prefix=row.get("assistant_prefix", ""),
        )
        batch["image"] = image
        batch = to_device(model, batch)
        seq_len = int(batch["input_ids"].shape[1])
        if pos >= seq_len:
            raise ValueError(f"source_pos_out_of_range pos={{pos}} seq_len={{seq_len}}")
        with torch.inference_mode():
            before = model.forward_from_batch(batch)
            wrong_id = top_wrong(before, target_id)
            before_target, before_rank = score_token(before, target_id)
            before_wrong, _ = score_token(before, wrong_id)
            after, _ = model.feature_intervention(
                batch,
                [(layer, pos, feature, 0.0)],
                freeze_attention=True,
                apply_activation_function=True,
                sparse=False,
            )
            after_target, after_rank = score_token(after, target_id)
            after_wrong, _ = score_token(after, wrong_id)
            source_damage = before_target - after_target
            source_wrong_damage = before_wrong - after_wrong
            control_damages = []
            for c_layer, c_pos, c_feature in parse_controls(row.get("control_nodes", "")):
                if c_pos >= seq_len:
                    continue
                c_after, _ = model.feature_intervention(
                    batch,
                    [(c_layer, c_pos, c_feature, 0.0)],
                    freeze_attention=True,
                    apply_activation_function=True,
                    sparse=False,
                )
                c_after_target, _ = score_token(c_after, target_id)
                control_damages.append(before_target - c_after_target)
        control_mean = sum(control_damages) / len(control_damages) if control_damages else 0.0
        base.update(
            {{
                "wrong_token_id": str(wrong_id),
                "source_damage_target_logit": f"{{source_damage:.10g}}",
                "source_damage_wrong_logit": f"{{source_wrong_damage:.10g}}",
                "source_minus_controls": f"{{(source_damage - control_mean):.10g}}",
                "correct_minus_wrong": f"{{(source_damage - source_wrong_damage):.10g}}",
                "control_damage_target_mean": f"{{control_mean:.10g}}",
                "control_count": str(len(control_damages)),
                "before_target_logit": f"{{before_target:.10g}}",
                "after_target_logit": f"{{after_target:.10g}}",
                "before_target_rank": str(before_rank),
                "after_target_rank": str(after_rank),
                "status": "ok",
                "error_message": "",
            }}
        )
        ok += 1
        print(f"[ok] {{idx}}/{{len(rows)}} {{row['candidate_id']}} damage={{source_damage:.4f}} gap={{source_damage-control_mean:.4f}}")
    except Exception as exc:
        base["status"] = "error"
        base["error_message"] = str(exc)
        errors += 1
        print(f"[error] {{idx}}/{{len(rows)}} {{row.get('candidate_id')}} {{exc}}")
    results.append(base)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
summary = {{
    "status": "ok" if ok else "blocked_no_success",
    "rows": len(rows),
    "ok": ok,
    "skipped": skipped,
    "errors": errors,
    "aligned_only": aligned_only,
    "elapsed_sec": time.time() - t0,
    "out_csv": str(out_csv),
}}
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo '--- Stage6 unified Gemma fixed-node done ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true
"""


def _fetch(sftp, mode: str, tag: str) -> None:
    for name in [f"{PREFIX}_{mode}_{tag}_manifest.csv", f"{PREFIX}_{mode}_{tag}_raw.csv", f"{PREFIX}_{mode}_{tag}_run.json"]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_CROSS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def _status_command(mode: str, tag: str) -> str:
    return f"""
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'stage6_unified_prompt_text|fixednode|gemma3-4b-it-plt' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{PREFIX}_{mode}_{tag}_* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*fixednode* 2>/dev/null || true
tail -n 80 {REMOTE_STAGE}/logs/*fixednode* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo DISK
df -h /root/autodl-tmp
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage6 unified Gemma fixed-node prompt probe.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="unified_v1")
    parser.add_argument("--aligned-only", action="store_true", default=True)
    parser.add_argument("--include-unaligned", action="store_false", dest="aligned_only")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    manifest = _manifest_for(args.tag)
    if not manifest.exists() and not args.status and not args.fetch_only:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "local" / "build_stage6_unified_metric_manifests.py"), "--tag", args.tag],
            check=True,
        )
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)

    if args.status:
        _stdin, stdout, stderr = client.exec_command(_status_command(args.mode, args.tag))
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch(sftp, args.mode, args.tag)
        sftp.close()
        client.close()
        if not args.skip_analyze:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "local" / "analyze_stage6_unified_prompt_text_metrics.py"),
                    "--tag",
                    args.tag,
                    "--mode",
                    args.mode,
                ],
                check=False,
            )
        return 0

    remote_manifest = _upload_manifest_and_assets(base, sftp, manifest, args.mode, args.tag)
    remote_script = f"{REMOTE_STAGE}/run_stage6_unified_fixednode_{args.mode}_{args.tag}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(
            _remote_script(
                mode=args.mode,
                tag=args.tag,
                remote_manifest=remote_manifest,
                aligned_only=args.aligned_only,
                max_rows=args.max_rows,
            ).replace("\r\n", "\n")
        )
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        remote_log_dir = f"{REMOTE_STAGE}/logs"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        remote_log = f"{remote_log_dir}/fixednode_{args.mode}_{stamp}.log"
        cmd = f"mkdir -p {remote_log_dir}; OUT={REMOTE_STAGE}/{PREFIX}_{args.mode}_{args.tag}_raw.csv SUMMARY={REMOTE_STAGE}/{PREFIX}_{args.mode}_{args.tag}_run.json nohup bash {remote_script} > {remote_log} 2>&1 < /dev/null & echo $!"
        _stdin, stdout, stderr = client.exec_command(cmd)
        pid = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        print({"detached_remote_pid": pid, "remote_log": remote_log, "remote_script": remote_script})
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    stdin, stdout, stderr = client.exec_command(
        f"OUT={REMOTE_STAGE}/{PREFIX}_{args.mode}_{args.tag}_raw.csv SUMMARY={REMOTE_STAGE}/{PREFIX}_{args.mode}_{args.tag}_run.json bash {remote_script}",
        get_pty=True,
    )
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
    _fetch(sftp, args.mode, args.tag)
    sftp.close()
    client.close()
    if exit_status == 0 and not args.skip_analyze:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "local" / "analyze_stage6_unified_prompt_text_metrics.py"),
                "--tag",
                args.tag,
                "--mode",
                args.mode,
            ],
            check=False,
        )
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
