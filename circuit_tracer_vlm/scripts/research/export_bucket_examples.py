#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-bucket sample examples for manual selection. "
            "Can optionally join manifest/question info and compare metrics."
        )
    )
    parser.add_argument("--bucket-csv", required=True, help="Input bucket csv from build_ab_bucket_csv.py")
    parser.add_argument("--out-csv", required=True, help="Output selected examples csv")
    parser.add_argument("--manifest-csv", default="", help="Optional manifest csv to join question/image_path")
    parser.add_argument("--compare-sample-csv", default="", help="Optional compare_prompt_ab_circuits sample csv")
    parser.add_argument("--per-bucket", type=int, default=20, help="Max samples per bucket")
    parser.add_argument(
        "--buckets",
        default="A1_B0,A0_B1",
        help="Comma list of buckets to export. Empty means all.",
    )
    parser.add_argument(
        "--sort-metric",
        default="",
        help="Optional metric column from compare-sample csv, e.g. delta_target_error_ratio",
    )
    parser.add_argument(
        "--sort-abs",
        action="store_true",
        default=True,
        help="Sort by absolute metric value (default true).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bucket_rows = _read_csv(Path(args.bucket_csv).expanduser().resolve())
    if not bucket_rows:
        raise ValueError("empty bucket csv")

    want_buckets = {b.strip() for b in args.buckets.split(",") if b.strip()} if args.buckets.strip() else set()

    manifest_map: dict[str, dict] = {}
    if args.manifest_csv:
        for r in _read_csv(Path(args.manifest_csv).expanduser().resolve()):
            sid = (r.get("sample_id") or "").strip()
            if sid:
                manifest_map[sid] = r

    metric_map: dict[str, dict] = {}
    if args.compare_sample_csv:
        for r in _read_csv(Path(args.compare_sample_csv).expanduser().resolve()):
            sid = (r.get("sample_id") or "").strip()
            if sid:
                metric_map[sid] = r

    grouped: dict[str, list[dict]] = {}
    for r in bucket_rows:
        sid = (r.get("sample_id") or "").strip()
        bkt = (r.get("bucket") or "").strip()
        if not sid or not bkt:
            continue
        if want_buckets and bkt not in want_buckets:
            continue
        merged = dict(r)
        if sid in manifest_map:
            merged["question"] = (manifest_map[sid].get("question") or "").strip()
            merged["image_path"] = (manifest_map[sid].get("image_path") or "").strip()
            merged["notes"] = (manifest_map[sid].get("notes") or "").strip()
        if sid in metric_map:
            for k, v in metric_map[sid].items():
                if k not in merged:
                    merged[k] = v
        grouped.setdefault(bkt, []).append(merged)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for bkt, rows in sorted(grouped.items(), key=lambda x: x[0]):
        if args.sort_metric:
            def _score(x: dict) -> float:
                v = _to_float(x.get(args.sort_metric, ""))
                return abs(v) if args.sort_abs else v

            rows = sorted(rows, key=_score, reverse=True)
        else:
            rows = rows[:]
            rng.shuffle(rows)
        take = rows[: max(1, args.per_bucket)]
        for i, r in enumerate(take, start=1):
            r["bucket_rank"] = str(i)
        selected.extend(take)

    if not selected:
        raise ValueError("no selected rows; check --buckets or input files")

    # Put key columns first; keep the rest.
    preferred = [
        "sample_id",
        "bucket",
        "bucket_rank",
        "a_correct",
        "b_correct",
        "a_pred",
        "a_gold",
        "b_pred",
        "b_gold",
        "question",
        "image_path",
        "delta_target_error_ratio",
        "delta_traced_max_depth",
        "delta_traced_nodes",
        "delta_traced_edges",
        "delta_replacement_score",
        "delta_completeness_score",
    ]
    seen = set()
    fields: list[str] = []
    for k in preferred:
        if any(k in r for r in selected):
            fields.append(k)
            seen.add(k)
    for r in selected:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)

    out_csv = Path(args.out_csv).expanduser().resolve()
    _write_csv(out_csv, selected, fields)
    print(f"[done] selected examples: {out_csv} rows={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
