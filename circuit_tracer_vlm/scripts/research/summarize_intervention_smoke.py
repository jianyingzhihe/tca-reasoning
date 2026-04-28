#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return math.nan
    return sum(clean) / len(clean)


def _frac(values: list[bool]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def _bucket_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("intervention_smoke_"):
        return stem[len("intervention_smoke_") :]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate intervention smoke CSVs into a compact summary.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more intervention_smoke_*.csv files.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    all_rows: list[dict[str, str]] = []
    for item in args.inputs:
        path = Path(item).expanduser().resolve()
        bucket_guess = _bucket_from_path(path)
        for row in _read_csv(path):
            if not row.get("bucket"):
                row["bucket"] = bucket_guess
            row["source_csv"] = str(path)
            all_rows.append(row)

    group_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        bucket = row.get("bucket", "") or "__unknown__"
        run = row.get("run", "") or "__unknown__"
        group_rows[(bucket, run)].append(row)
    group_rows[("__all__", "__all__")] = all_rows

    summary_rows: list[dict[str, str]] = []
    strongest_rows: list[dict[str, str]] = []

    for (bucket, run), rows in sorted(group_rows.items()):
        logit_deltas = [_safe_float(r.get("delta_target_logit")) for r in rows]
        prob_deltas = [_safe_float(r.get("delta_target_prob")) for r in rows]
        path_mass = [_safe_float(r.get("path_mass_best")) for r in rows]

        valid_logit = [v for v in logit_deltas if not math.isnan(v)]
        valid_prob = [v for v in prob_deltas if not math.isnan(v)]
        mean_logit = _mean(logit_deltas)
        mean_prob = _mean(prob_deltas)
        frac_neg_logit = _frac([v < 0 for v in valid_logit])
        frac_neg_prob = _frac([v < 0 for v in valid_prob])

        summary_rows.append(
            {
                "bucket": bucket,
                "run": run,
                "n_interventions": str(len(rows)),
                "mean_delta_target_logit": _fmt(mean_logit),
                "mean_delta_target_prob": _fmt(mean_prob),
                "frac_negative_delta_target_logit": _fmt(frac_neg_logit),
                "frac_negative_delta_target_prob": _fmt(frac_neg_prob),
                "mean_path_mass_best": _fmt(_mean(path_mass)),
            }
        )

        strongest = None
        strongest_value = math.inf
        for row in rows:
            delta = _safe_float(row.get("delta_target_logit"))
            if math.isnan(delta):
                continue
            if delta < strongest_value:
                strongest_value = delta
                strongest = row
        if strongest is not None:
            strongest_rows.append(
                {
                    "bucket": bucket,
                    "run": run,
                    "sample_id": strongest.get("sample_id", ""),
                    "feature_layer": strongest.get("feature_layer", ""),
                    "feature_pos": strongest.get("feature_pos", ""),
                    "feature_id": strongest.get("feature_id", ""),
                    "path_mass_best": strongest.get("path_mass_best", ""),
                    "delta_target_logit": strongest.get("delta_target_logit", ""),
                    "delta_target_prob": strongest.get("delta_target_prob", ""),
                    "top1_before_token": strongest.get("top1_before_token", ""),
                    "top1_after_token": strongest.get("top1_after_token", ""),
                    "source_csv": strongest.get("source_csv", ""),
                }
            )

    _write_csv(
        out_dir / "intervention_smoke_summary.csv",
        summary_rows,
        [
            "bucket",
            "run",
            "n_interventions",
            "mean_delta_target_logit",
            "mean_delta_target_prob",
            "frac_negative_delta_target_logit",
            "frac_negative_delta_target_prob",
            "mean_path_mass_best",
        ],
    )
    _write_csv(
        out_dir / "intervention_smoke_strongest_rows.csv",
        strongest_rows,
        [
            "bucket",
            "run",
            "sample_id",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "path_mass_best",
            "delta_target_logit",
            "delta_target_prob",
            "top1_before_token",
            "top1_after_token",
            "source_csv",
        ],
    )

    lines = [
        "# Intervention Smoke Summary",
        "",
        "A more negative `delta_target_logit` or `delta_target_prob` means zeroing that feature hurt the target token more.",
        "",
        "## Aggregated Metrics",
        "",
        "| bucket | run | n | mean dlogit | mean dprob | frac dlogit<0 | frac dprob<0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {bucket} | {run} | {n} | {dlogit} | {dprob} | {fneglogit} | {fnegprob} |".format(
                bucket=row["bucket"],
                run=row["run"],
                n=row["n_interventions"],
                dlogit=row["mean_delta_target_logit"],
                dprob=row["mean_delta_target_prob"],
                fneglogit=row["frac_negative_delta_target_logit"],
                fnegprob=row["frac_negative_delta_target_prob"],
            )
        )

    if strongest_rows:
        lines.extend(
            [
                "",
                "## Strongest Single-Feature Drops",
                "",
                "| bucket | run | sample_id | feature | delta_target_logit | delta_target_prob |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for row in strongest_rows:
            lines.append(
                "| {bucket} | {run} | {sample} | L{layer}:P{pos}:F{feature} | {dlogit} | {dprob} |".format(
                    bucket=row["bucket"],
                    run=row["run"],
                    sample=row["sample_id"],
                    layer=row["feature_layer"],
                    pos=row["feature_pos"],
                    feature=row["feature_id"],
                    dlogit=row["delta_target_logit"],
                    dprob=row["delta_target_prob"],
                )
            )

    (out_dir / "intervention_smoke_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
