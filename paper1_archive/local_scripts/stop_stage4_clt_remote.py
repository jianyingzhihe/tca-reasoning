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
set +e
PIDS=$(ps -eo pid,cmd | grep -E 'run_stage4_clt_finalization|run_stage2o_attribution_weighted_feature_bridge|run_stage2o_cross_model_source_control_probe' | grep -v grep | awk '{print $1}')
if [ -n "$PIDS" ]; then
  echo "stopping: $PIDS"
  kill -TERM $PIDS 2>/dev/null || true
  sleep 2
  kill -KILL $PIDS 2>/dev/null || true
else
  echo "no stage4 clt foreground process found"
fi
echo 'remaining:'
ps -eo pid,etime,cmd | grep -E 'run_stage4_clt_finalization|run_stage2o_attribution_weighted_feature_bridge|run_stage2o_cross_model_source_control_probe' | grep -v grep || true
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

