#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_summary_line(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or "=" not in s:
                continue
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _read_compare_all_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if (r.get("bucket") or "").strip() == "__all__":
            return r
    return rows[-1] if rows else {}


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run 3-prompt comparison by launching pairwise AB/AC/BC bucket+compare pipelines. "
            "Outputs per-pair files and a merged overview csv."
        )
    )
    parser.add_argument("--pt-dir-a", required=True)
    parser.add_argument("--pt-dir-b", required=True)
    parser.add_argument("--pt-dir-c", required=True)
    parser.add_argument("--eval-a-csv", required=True)
    parser.add_argument("--eval-b-csv", required=True)
    parser.add_argument("--eval-c-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--correct-col", default="")
    parser.add_argument("--pred-col", default="")
    parser.add_argument("--gold-col", default="")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--require-same-pred",
        action="store_true",
        help="For each pair, keep only rows where normalized predictions are identical.",
    )
    parser.add_argument("--target-logit-rank", type=int, default=0)
    parser.add_argument("--parents-per-node", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=48)
    parser.add_argument("--min-abs-weight", type=float, default=0.0)
    parser.add_argument("--skip-graph-scores", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    bucket_dir = out_dir / "buckets"
    pair_dir = out_dir / "pairwise"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    bucket_script = here / "build_ab_bucket_csv.py"
    compare_script = here / "compare_prompt_ab_circuits.py"
    py = sys.executable

    runs = {
        "A": {"pt": str(Path(args.pt_dir_a).expanduser().resolve()), "eval": str(Path(args.eval_a_csv).expanduser().resolve())},
        "B": {"pt": str(Path(args.pt_dir_b).expanduser().resolve()), "eval": str(Path(args.eval_b_csv).expanduser().resolve())},
        "C": {"pt": str(Path(args.pt_dir_c).expanduser().resolve()), "eval": str(Path(args.eval_c_csv).expanduser().resolve())},
    }
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]

    overview_rows: list[dict] = []
    for left, right in pairs:
        pair_name = f"{left}{right}"
        bucket_csv = bucket_dir / f"{pair_name}_bucket.csv"
        bucket_summary = bucket_dir / f"{pair_name}_bucket_summary.txt"
        compare_sample = pair_dir / f"{pair_name}_circuit_sample.csv"
        compare_summary = pair_dir / f"{pair_name}_circuit_summary.csv"

        cmd_bucket = [
            py,
            str(bucket_script),
            "--run-a-csv",
            runs[left]["eval"],
            "--run-b-csv",
            runs[right]["eval"],
            "--out-bucket-csv",
            str(bucket_csv),
            "--out-summary-txt",
            str(bucket_summary),
            "--id-col",
            args.id_col,
        ]
        if args.correct_col:
            cmd_bucket.extend(["--correct-col", args.correct_col])
        if args.pred_col:
            cmd_bucket.extend(["--pred-col", args.pred_col])
        if args.gold_col:
            cmd_bucket.extend(["--gold-col", args.gold_col])
        if args.no_normalize:
            cmd_bucket.append("--no-normalize")
        if args.require_same_pred:
            cmd_bucket.append("--require-same-pred")
        _run(cmd_bucket)

        cmd_compare = [
            py,
            str(compare_script),
            "--pt-dir-a",
            runs[left]["pt"],
            "--pt-dir-b",
            runs[right]["pt"],
            "--bucket-csv",
            str(bucket_csv),
            "--out-sample-csv",
            str(compare_sample),
            "--out-summary-csv",
            str(compare_summary),
            "--target-logit-rank",
            str(args.target_logit_rank),
            "--parents-per-node",
            str(args.parents_per_node),
            "--max-depth",
            str(args.max_depth),
            "--min-abs-weight",
            str(args.min_abs_weight),
            "--log-every",
            str(args.log_every),
        ]
        if args.skip_graph_scores:
            cmd_compare.append("--skip-graph-scores")
        _run(cmd_compare)

        bsum = _read_summary_line(bucket_summary)
        csum = _read_compare_all_row(compare_summary)
        overview_rows.append(
            {
                "pair": pair_name,
                "left_prompt": left,
                "right_prompt": right,
                "rows": bsum.get("rows", ""),
                "left_accuracy": bsum.get("a_accuracy", ""),
                "right_accuracy": bsum.get("b_accuracy", ""),
                "A1_B1": bsum.get("A1_B1", ""),
                "A1_B0": bsum.get("A1_B0", ""),
                "A0_B1": bsum.get("A0_B1", ""),
                "A0_B0": bsum.get("A0_B0", ""),
                "node_overlap_jaccard": csum.get("node_overlap_jaccard", ""),
                "edge_overlap_jaccard": csum.get("edge_overlap_jaccard", ""),
                "delta_target_error_ratio": csum.get("delta_target_error_ratio", ""),
                "delta_traced_max_depth": csum.get("delta_traced_max_depth", ""),
                "delta_traced_nodes": csum.get("delta_traced_nodes", ""),
                "delta_traced_edges": csum.get("delta_traced_edges", ""),
                "delta_replacement_score": csum.get("delta_replacement_score", ""),
                "delta_completeness_score": csum.get("delta_completeness_score", ""),
            }
        )

    overview_csv = out_dir / "prompt_abc_overview.csv"
    _write_csv(
        overview_csv,
        overview_rows,
        [
            "pair",
            "left_prompt",
            "right_prompt",
            "rows",
            "left_accuracy",
            "right_accuracy",
            "A1_B1",
            "A1_B0",
            "A0_B1",
            "A0_B0",
            "node_overlap_jaccard",
            "edge_overlap_jaccard",
            "delta_target_error_ratio",
            "delta_traced_max_depth",
            "delta_traced_nodes",
            "delta_traced_edges",
            "delta_replacement_score",
            "delta_completeness_score",
        ],
    )
    print(f"[done] overview: {overview_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
