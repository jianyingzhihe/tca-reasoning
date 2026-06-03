#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(r"E:\Bridging")


def main() -> int:
    # The validation runner intentionally reuses the Stage5 layer-screen runner.
    # Validation configs are selected by passing explicit --layers/--topks from
    # the near-pass table; strict confirmation uses --pack strict and
    # --mode strict-confirm.
    script = ROOT / "scripts" / "local" / "run_stage5_clt_layer_screen_remote.py"
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(script)}
    code = script.read_text(encoding="utf-8")
    sys.argv = [str(script), *sys.argv[1:]]
    exec(compile(code, str(script), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
