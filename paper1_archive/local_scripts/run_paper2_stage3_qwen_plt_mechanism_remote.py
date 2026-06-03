#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")
BASE_RUNNER = ROOT / "scripts" / "local" / "run_stage4_qwen_plt_layer_sweep_remote.py"
STAGE3 = ROOT / "doc" / "experiments" / "paper2" / "stage3_mechanism"
PREFIX = "paper2_stage3_qwen_plt_mechanism"


def _load_base():
    spec = importlib.util.spec_from_file_location("stage4_qwen_plt_layer_sweep_remote", BASE_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_cli(argv: list[str]) -> list[str]:
    out: list[str] = []
    seen = {"--mode": False, "--packs": False, "--layers": False, "--max-prompt-runs": False, "--top-per-prompt-run": False, "--main-per-prompt-run": False}
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in seen:
            seen[arg] = True
        out.append(arg)
        idx += 1
    if not seen["--mode"]:
        out.extend(["--mode", "smoke"])
    if not seen["--packs"]:
        out.extend(["--packs", "primary"])
    if not seen["--layers"]:
        out.extend(["--layers", "13,15,17"])
    if not seen["--max-prompt-runs"]:
        out.extend(["--max-prompt-runs", "0"])
    if not seen["--top-per-prompt-run"]:
        out.extend(["--top-per-prompt-run", "8"])
    if not seen["--main-per-prompt-run"]:
        out.extend(["--main-per-prompt-run", "4"])
    return out


def main() -> int:
    tag = "paper2_mechanism_v1"
    if "--tag" in sys.argv:
        idx = sys.argv.index("--tag")
        if idx + 1 < len(sys.argv):
            tag = sys.argv[idx + 1]
            del sys.argv[idx : idx + 2]
    runs = STAGE3 / f"paper2_stage3_mechanism_qwen_prompt_runs_{tag}.csv"
    if not runs.exists():
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "local" / "build_paper2_stage3_mechanism_pack.py"), "--tag", tag], check=True)
    if not runs.exists():
        raise FileNotFoundError(runs)

    base = _load_base()
    base.REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/paper2_stage3_qwen_plt_mechanism"
    base.REMOTE_ASSETS = f"{base.REMOTE_STAGE}/assets_{tag}"
    base.LOCAL_CROSS = STAGE3
    base.PRIMARY_RUNS = runs
    base.STRICT_RUNS = runs

    def stem(pack: str, mode: str, layer: int, artifact: str) -> str:
        return f"{PREFIX}_{pack}_{mode}_L{layer}_{tag}_{artifact}"

    def status_command(mode: str, packs: list[str], layers: list[int]) -> str:
        patterns = "|".join(stem(pack, mode, layer, "")[:-1] for pack in packs for layer in layers)
        return f"""
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'paper2_stage3_qwen_plt_mechanism|run_stage4_qwen_evidence_first|run_stage4_qwen_causal_cutter|{patterns}' | grep -v grep || true
echo FILES
ls -lh {base.REMOTE_STAGE}/{PREFIX}_*_{mode}_*_{tag}_* 2>/dev/null || true
echo LOGS
ls -lh {base.REMOTE_STAGE}/logs/* 2>/dev/null || true
tail -n 80 {base.REMOTE_STAGE}/logs/* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo DISK
df -h /root/autodl-tmp
"""

    base._stem = stem
    base._status_command = status_command
    sys.argv = [str(BASE_RUNNER), *_parse_cli(sys.argv[1:]), "--skip-analyze"]
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
