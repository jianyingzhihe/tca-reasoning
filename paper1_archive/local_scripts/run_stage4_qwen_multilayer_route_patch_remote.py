#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(r"E:\Bridging")
CROSS = ROOT / "doc" / "experiments" / "stage4" / "cross_model"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4-024 multi-layer route patch entrypoint.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--decision-json", type=Path, default=CROSS / "stage4_qwen_decisive_route_multilayer_decision.json")
    args = parser.parse_args()
    decision = {
        "status": "blocked",
        "reason": "needs_hidden_lattice_and_plt_layer_sweep",
        "mode": args.mode,
        "next_step": "Use top hidden layers and their PLT candidates to run simultaneous multi-layer patching.",
    }
    args.decision_json.parent.mkdir(parents=True, exist_ok=True)
    args.decision_json.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
