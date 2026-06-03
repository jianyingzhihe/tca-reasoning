#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


BUCKET_ORDER = ["A0_B0", "A0_B1", "A1_B0", "A1_B1"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _priority_score(row: dict[str, str]) -> tuple[int, int, int, int]:
    visual = (row.get("visual_type_label") or "").strip()
    knowledge = (row.get("knowledge_level_label") or "").strip()
    priority = (row.get("priority") or "").strip()
    rank_str = (row.get("rank") or "").strip()
    rank = int(rank_str) if rank_str.isdigit() else 999

    score = 0
    if knowledge == "medium":
        score += 4
    if visual == "diffuse_global":
        score += 3
    if visual == "multi_region":
        score += 2
    if knowledge == "low":
        score += 1
    priority_bonus = {"high": 0, "medium": 1, "low": 2}.get(priority, 3)
    return (-score, priority_bonus, rank, hash(row.get("item_id", "")) % 1000000)


def _join_rows(manifest_rows: list[dict[str, str]], label_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    labels_by_item = {(row.get("item_id") or "").strip(): row for row in label_rows}
    out = []
    for row in manifest_rows:
        item_id = (row.get("item_id") or "").strip()
        merged = dict(row)
        merged.update(labels_by_item.get(item_id, {}))
        out.append(merged)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a larger targeted Stage 1 sample-id pack from dual type labels.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-per-bucket", type=int, default=8)
    args = parser.parse_args()

    manifest_rows = _read_csv(Path(args.manifest_csv).expanduser().resolve())
    label_rows = _read_csv(Path(args.labels_csv).expanduser().resolve())
    rows = _join_rows(manifest_rows, label_rows)

    filtered = []
    for row in rows:
        visual = (row.get("visual_type_label") or "").strip()
        knowledge = (row.get("knowledge_level_label") or "").strip()
        if knowledge == "medium" or knowledge == "low" or visual in {"diffuse_global", "multi_region"}:
            filtered.append(row)

    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in filtered:
        by_bucket[(row.get("bucket") or "").strip()].append(row)

    out_dir = Path(args.out_dir).expanduser().resolve()
    sample_ids_dir = out_dir / "bucket_sample_ids"
    sample_ids_dir.mkdir(parents=True, exist_ok=True)

    chosen_rows: list[dict[str, str]] = []
    for bucket in BUCKET_ORDER:
        bucket_rows = by_bucket.get(bucket, [])
        bucket_rows.sort(key=_priority_score)
        picked = bucket_rows[: args.max_per_bucket]
        _write_csv(
            sample_ids_dir / f"{bucket}.csv",
            [{"sample_id": row.get("sample_id", "")} for row in picked],
            ["sample_id"],
        )
        for order_idx, row in enumerate(picked, start=1):
            chosen_rows.append(
                {
                    "bucket": bucket,
                    "bucket_rank": str(order_idx),
                    "sample_id": row.get("sample_id", ""),
                    "visual_type_label": row.get("visual_type_label", ""),
                    "knowledge_level_label": row.get("knowledge_level_label", ""),
                    "priority": row.get("priority", ""),
                    "rank": row.get("rank", ""),
                    "display_question": row.get("display_question", ""),
                    "answer_text": row.get("answer_text", ""),
                }
            )

    _write_csv(
        out_dir / "targeted_type_label_samples.csv",
        chosen_rows,
        [
            "bucket",
            "bucket_rank",
            "sample_id",
            "visual_type_label",
            "knowledge_level_label",
            "priority",
            "rank",
            "display_question",
            "answer_text",
        ],
    )

    md_lines = [
        "# Targeted Type-Label Sample Pack",
        "",
        f"- max per bucket: `{args.max_per_bucket}`",
        "",
        "| bucket | bucket rank | sample_id | visual | knowledge | priority | rank |",
        "|---|---:|---|---|---|---|---:|",
    ]
    for row in chosen_rows:
        md_lines.append(
            "| {bucket} | {bucket_rank} | {sample_id} | {visual_type_label} | {knowledge_level_label} | {priority} | {rank} |".format(
                **row
            )
        )
    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[done] out_dir={out_dir}")
    print(f"[done] total_selected={len(chosen_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
