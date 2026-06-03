#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


AMBIGUITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(text: str) -> float:
    try:
        return float((text or "").strip() or "0")
    except ValueError:
        return 0.0


def _safe_int(text: str) -> int:
    try:
        return int((text or "").strip() or "0")
    except ValueError:
        return 0


def _ambiguity_rank(text: str) -> int:
    return AMBIGUITY_ORDER.get((text or "").strip(), 9)


def _load_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id:
                excluded.add(sample_id)
    return excluded


def _round_robin_by_op(rows: list[dict[str, str]], ops: list[str], target_size: int) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["reasoning_operation"]].append(row)

    for op_rows in groups.values():
        op_rows.sort(
            key=lambda row: (
                -_safe_int(row.get("has_behavior_disagreement", "0")),
                -abs(_safe_float(row.get("delta_vqa_score", "0"))),
                _ambiguity_rank(row.get("ambiguity_flag", "")),
                row.get("sample_id", ""),
            )
        )

    selected: list[dict[str, str]] = []
    used: set[str] = set()
    made_progress = True
    while len(selected) < target_size and made_progress:
        made_progress = False
        for op in ops:
            queue = groups.get(op, [])
            while queue and queue[0].get("sample_id", "") in used:
                queue.pop(0)
            if not queue:
                continue
            row = queue.pop(0)
            sample_id = row.get("sample_id", "")
            if not sample_id or sample_id in used:
                continue
            selected.append(row)
            used.add(sample_id)
            made_progress = True
            if len(selected) >= target_size:
                break
    return selected


def _manifest_subset(manifest_rows: list[dict[str, str]], selected_ids: set[str]) -> list[dict[str, str]]:
    rows = [row for row in manifest_rows if (row.get("sample_id") or "").strip() in selected_ids]
    rows.sort(key=lambda row: (row.get("sample_id") or ""))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a larger B/D visual-positive held-out pack from the scored candidate pool."
    )
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--manifest-b-csv", required=True)
    parser.add_argument("--manifest-d-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True, help="Output name prefix, e.g. visual_positive_strong12")
    parser.add_argument("--target-size", type=int, default=12)
    parser.add_argument(
        "--ops",
        default="scene_inference,symbol_text_reading,visual_readout",
        help="Comma-separated reasoning_operation values to keep.",
    )
    parser.add_argument("--strong-only", action="store_true")
    parser.add_argument("--exclude-csv", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = [part.strip() for part in args.ops.split(",") if part.strip()]
    candidates = _read_csv(Path(args.candidates_csv).expanduser().resolve())
    manifest_b = _read_csv(Path(args.manifest_b_csv).expanduser().resolve())
    manifest_d = _read_csv(Path(args.manifest_d_csv).expanduser().resolve())
    excluded_ids = _load_excluded_ids([Path(p).expanduser().resolve() for p in args.exclude_csv])

    filtered: list[dict[str, str]] = []
    for row in candidates:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id or sample_id in excluded_ids:
            continue
        if (row.get("eligible") or "").strip() != "1":
            continue
        if (row.get("reasoning_operation") or "").strip() not in ops:
            continue
        if args.strong_only and (row.get("image_dependence_group") or "").strip() != "strong":
            continue
        filtered.append(row)

    selected = _round_robin_by_op(filtered, ops=ops, target_size=args.target_size)
    selected_ids = {row["sample_id"] for row in selected}

    selected_fieldnames = [
        "sample_id",
        "item_id",
        "bucket",
        "priority",
        "source_round",
        "question_text",
        "answer_text",
        "question_type",
        "visual_structure",
        "image_dependence",
        "image_dependence_group",
        "reasoning_operation",
        "ambiguity_flag",
        "extra_evidence",
        "annotator_notes",
        "B_strict",
        "D_strict",
        "B_vqa_score",
        "D_vqa_score",
        "delta_strict",
        "delta_vqa_score",
        "has_behavior_disagreement",
        "B_nonempty",
        "D_nonempty",
        "B_format_ok",
        "D_format_ok",
        "eligible",
        "B_predicted_answer",
        "D_predicted_answer",
    ]

    sample_id_rows = [
        {
            "sample_id": row["sample_id"],
            "image_dependence_group": row.get("image_dependence_group", ""),
            "reasoning_operation": row.get("reasoning_operation", ""),
            "visual_structure": row.get("visual_structure", ""),
        }
        for row in selected
    ]

    manifest_b_subset = _manifest_subset(manifest_b, selected_ids)
    manifest_d_subset = _manifest_subset(manifest_d, selected_ids)

    sample_ids_path = out_dir / f"{args.tag}_sample_ids.csv"
    selected_path = out_dir / f"{args.tag}_selected.csv"
    manifest_b_path = out_dir / f"manifest_B_direct_{args.tag}.csv"
    manifest_d_path = out_dir / f"manifest_D_visual_only_{args.tag}.csv"
    summary_path = out_dir / f"{args.tag}_summary.md"

    _write_csv(sample_ids_path, sample_id_rows, ["sample_id", "image_dependence_group", "reasoning_operation", "visual_structure"])
    _write_csv(selected_path, selected, selected_fieldnames)
    if manifest_b_subset:
        _write_csv(manifest_b_path, manifest_b_subset, list(manifest_b_subset[0].keys()))
    if manifest_d_subset:
        _write_csv(manifest_d_path, manifest_d_subset, list(manifest_d_subset[0].keys()))

    counts = Counter(row["reasoning_operation"] for row in selected)
    disagree = Counter(row["reasoning_operation"] for row in selected if _safe_int(row.get("has_behavior_disagreement", "0")) == 1)
    lines = [
        f"# {args.tag}",
        "",
        f"- target size: `{args.target_size}`",
        f"- selected size: `{len(selected)}`",
        f"- strong only: `{args.strong_only}`",
        f"- reasoning ops: `{', '.join(ops)}`",
        "",
        "## Composition",
        "",
    ]
    for op in ops:
        lines.append(f"- `{op}`: `{counts.get(op, 0)}` selected, `{disagree.get(op, 0)}` with behavior disagreement")
    lines.extend(["", "## Selected rows", ""])
    for row in selected:
        lines.append(
            "- "
            f"`{row['sample_id']}` / `{row.get('reasoning_operation', '')}` / "
            f"`{row.get('visual_structure', '')}` / "
            f"`B={row.get('B_strict', '')}` / `D={row.get('D_strict', '')}` / "
            f"`delta_vqa={row.get('delta_vqa_score', '')}`"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
