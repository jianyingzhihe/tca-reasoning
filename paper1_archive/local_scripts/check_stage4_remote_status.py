#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner", ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    base = _load_base_runner()
    sys.path.insert(0, str(ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    cmd = r"""
set -e
STAGE=/root/autodl-tmp/tca-reasoning/stage4_qwen_source_tracing
echo '--- PS ---'
ps -eo pid,etime,cmd | grep -E 'stage4_qwen|run_qwen_answer|trace_compare|intervention' | grep -v grep || true
echo '--- DISK ---'
df -h /root/autodl-tmp || true
echo '--- TOP FILES ---'
ls -lh "$STAGE" 2>/dev/null | tail -80 || true
echo '--- GRAPH COUNTS ---'
find "$STAGE/source_tracing_primary_full/graphs_a_D_visual_only" -name '*.pt' 2>/dev/null | wc -l
find "$STAGE/source_tracing_primary_full/graphs_b_B_direct" -name '*.pt' 2>/dev/null | wc -l
echo '--- KEY ARTIFACTS ---'
for f in \
  "$STAGE/stage4_qwen_source_tracing_primary_full_meta_a.csv" \
  "$STAGE/stage4_qwen_source_tracing_primary_full_meta_b.csv" \
  "$STAGE/stage4_qwen_source_tracing_primary_full_sample_compare_controlled.csv" \
  "$STAGE/stage4_qwen_source_tracing_primary_full_nodes_detailed_controlled.csv" \
  "$STAGE/stage4_qwen_source_tracing_primary_full_intervention.csv" \
  "$STAGE/stage4_qwen_source_tracing_primary_full_decision.json"; do
  if [ -f "$f" ]; then
    ls -lh "$f"
  else
    echo "missing $f"
  fi
done
"""
    _stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        print(err)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

