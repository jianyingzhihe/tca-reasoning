#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Phase A+B workspace files/folders.")
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tpl = root / "research" / "templates"
    work = root / "research" / "work"
    out = root / "outputs" / "phase_ab"

    for d in [
        work,
        out / "raw_pt",
        out / "graph_json",
        out / "metrics",
        out / "figures",
        out / "logs",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    copy_if_missing(tpl / "repro_config.yaml", work / "repro_config.yaml")
    copy_if_missing(tpl / "sample_manifest.csv", work / "sample_manifest.csv")
    copy_if_missing(tpl / "dataset_with_trace.example.jsonl", work / "dataset_with_trace.jsonl")
    copy_if_missing(tpl / "step_alignment_template.csv", work / "step_alignment.csv")

    print(f"[setup_phase_ab] initialized at: {root}")
    print(f"[setup_phase_ab] work dir: {work}")
    print(f"[setup_phase_ab] output dir: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

