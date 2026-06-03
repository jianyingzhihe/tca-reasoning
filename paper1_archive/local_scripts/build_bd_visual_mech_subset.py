from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ANSWER_PREFIX = "the answer is "
DEP_GROUP_ORDER = ["strong", "mixed_or_weak"]
OP_ORDER = [
    "entity_linking",
    "world_fact_retrieval",
    "scene_inference",
    "symbol_text_reading",
    "commonsense_affordance",
    "visual_readout",
    "relation_reasoning",
]
AMBIGUITY_ORDER = {"low": 0, "medium": 1, "high": 2}
SOURCE_ROUND_ORDER = {"new320": 0, "old80": 1}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _truthy_answer(text: str) -> bool:
    return bool((text or "").strip())


def _format_ok(text: str) -> bool:
    return (text or "").strip().lower().startswith(ANSWER_PREFIX)


def _safe_int(text: str) -> int:
    try:
        return int((text or "").strip() or "0")
    except ValueError:
        return 0


def _safe_float(text: str) -> float:
    try:
        return float((text or "").strip() or "0")
    except ValueError:
        return 0.0


def _dep_group(text: str) -> str:
    return "strong" if (text or "").strip() == "strong" else "mixed_or_weak"


def _ambiguity_rank(text: str) -> int:
    return AMBIGUITY_ORDER.get((text or "").strip(), 9)


def _source_round_rank(text: str) -> int:
    return SOURCE_ROUND_ORDER.get((text or "").strip(), 9)


def _load_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        rows = _read_csv(path)
        for row in rows:
            sid = (row.get("sample_id") or "").strip()
            if sid:
                excluded.add(sid)
    return excluded


def _round_robin_select(
    candidates: list[dict[str, str]],
    *,
    target_size: int,
) -> list[dict[str, str]]:
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        strata[(row["image_dependence_group"], row["reasoning_operation"])].append(row)

    for values in strata.values():
        values.sort(
            key=lambda row: (
                -int(row["has_behavior_disagreement"]),
                -abs(float(row["delta_vqa_score"])),
                -abs(int(row["delta_strict"])),
                _ambiguity_rank(row["ambiguity_flag"]),
                _source_round_rank(row["source_round"]),
                row["sample_id"],
            )
        )

    stratum_order: list[tuple[str, str]] = []
    for dep in DEP_GROUP_ORDER:
        for op in OP_ORDER:
            key = (dep, op)
            if key in strata:
                stratum_order.append(key)
    for key in sorted(strata):
        if key not in stratum_order:
            stratum_order.append(key)

    selected: list[dict[str, str]] = []
    used: set[str] = set()
    made_progress = True
    while len(selected) < target_size and made_progress:
        made_progress = False
        for key in stratum_order:
            queue = strata.get(key, [])
            while queue and queue[0]["sample_id"] in used:
                queue.pop(0)
            if not queue:
                continue
            row = queue.pop(0)
            sid = row["sample_id"]
            if sid in used:
                continue
            selected.append(row)
            used.add(sid)
            made_progress = True
            if len(selected) >= target_size:
                break
    return selected


def _manifest_subset(
    manifest_rows: list[dict[str, str]],
    selected_ids: set[str],
) -> list[dict[str, str]]:
    rows = [row for row in manifest_rows if (row.get("sample_id") or "").strip() in selected_ids]
    rows.sort(key=lambda row: (row.get("sample_id") or ""))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a held-out B_direct vs D_visual_only mechanism subset from the labeled 395-pack."
    )
    parser.add_argument("--merged-csv", required=True)
    parser.add_argument("--manifest-b-csv", required=True)
    parser.add_argument("--manifest-d-csv", required=True)
    parser.add_argument("--exclude-csv", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--main-size", type=int, default=24)
    parser.add_argument("--sanity-size", type=int, default=5)
    args = parser.parse_args()

    merged_rows = _read_csv(Path(args.merged_csv).expanduser().resolve())
    manifest_b_rows = _read_csv(Path(args.manifest_b_csv).expanduser().resolve())
    manifest_d_rows = _read_csv(Path(args.manifest_d_csv).expanduser().resolve())
    exclude_paths = [Path(p).expanduser().resolve() for p in args.exclude_csv]
    excluded_ids = _load_excluded_ids(exclude_paths)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_b_by_id = {(row.get("sample_id") or "").strip(): row for row in manifest_b_rows}
    manifest_d_by_id = {(row.get("sample_id") or "").strip(): row for row in manifest_d_rows}

    candidate_rows: list[dict[str, str]] = []
    for row in merged_rows:
        sid = (row.get("sample_id") or "").strip()
        if not sid or sid in excluded_ids:
            continue
        if sid not in manifest_b_by_id or sid not in manifest_d_by_id:
            continue

        b_text = row.get("B_direct__generated_text", "")
        d_text = row.get("D_visual_only__generated_text", "")
        b_answer = row.get("B_direct__predicted_answer", "")
        d_answer = row.get("D_visual_only__predicted_answer", "")
        ambiguity_flag = (row.get("ambiguity_flag") or "").strip()
        reasoning_operation = (row.get("reasoning_operation") or "").strip()
        image_dependence = (row.get("image_dependence") or "").strip()

        b_nonempty = _truthy_answer(b_text) and _truthy_answer(b_answer)
        d_nonempty = _truthy_answer(d_text) and _truthy_answer(d_answer)
        b_format_ok = _format_ok(b_text)
        d_format_ok = _format_ok(d_text)

        eligible = (
            ambiguity_flag != "high"
            and bool(reasoning_operation)
            and bool(image_dependence)
            and b_nonempty
            and d_nonempty
            and b_format_ok
            and d_format_ok
        )

        b_strict = _safe_int(row.get("B_direct__correct", "0"))
        d_strict = _safe_int(row.get("D_visual_only__correct", "0"))
        b_vqa = _safe_float(row.get("B_direct__vqa_score", "0"))
        d_vqa = _safe_float(row.get("D_visual_only__vqa_score", "0"))

        candidate_rows.append(
            {
                "sample_id": sid,
                "item_id": (row.get("item_id") or "").strip(),
                "bucket": (row.get("bucket") or "").strip(),
                "priority": (row.get("priority") or "").strip(),
                "source_round": (row.get("source_round") or "").strip(),
                "question_text": (row.get("question_text") or "").strip(),
                "answer_text": (row.get("answer_text") or "").strip(),
                "question_type": (row.get("question_type") or "").strip(),
                "visual_structure": (row.get("visual_structure") or "").strip(),
                "image_dependence": image_dependence,
                "image_dependence_group": _dep_group(image_dependence),
                "reasoning_operation": reasoning_operation,
                "ambiguity_flag": ambiguity_flag,
                "extra_evidence": (row.get("extra_evidence") or "").strip(),
                "annotator_notes": (row.get("annotator_notes") or "").strip(),
                "B_strict": str(b_strict),
                "D_strict": str(d_strict),
                "B_vqa_score": f"{b_vqa:.6f}",
                "D_vqa_score": f"{d_vqa:.6f}",
                "delta_strict": str(d_strict - b_strict),
                "delta_vqa_score": f"{(d_vqa - b_vqa):.6f}",
                "has_behavior_disagreement": "1" if b_strict != d_strict else "0",
                "B_nonempty": "1" if b_nonempty else "0",
                "D_nonempty": "1" if d_nonempty else "0",
                "B_format_ok": "1" if b_format_ok else "0",
                "D_format_ok": "1" if d_format_ok else "0",
                "eligible": "1" if eligible else "0",
                "B_predicted_answer": (row.get("B_direct__predicted_answer") or "").strip(),
                "D_predicted_answer": (row.get("D_visual_only__predicted_answer") or "").strip(),
            }
        )

    eligible_rows = [row for row in candidate_rows if row["eligible"] == "1"]
    selected_main = _round_robin_select(eligible_rows, target_size=args.main_size)
    selected_main_ids = {row["sample_id"] for row in selected_main}
    selected_sanity = selected_main[: args.sanity_size]
    selected_sanity_ids = {row["sample_id"] for row in selected_sanity}

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

    _write_csv(out_dir / "all_candidates_scored.csv", candidate_rows, selected_fieldnames)
    _write_csv(out_dir / "selected_main.csv", selected_main, selected_fieldnames)
    _write_csv(out_dir / "selected_sanity.csv", selected_sanity, selected_fieldnames)

    manifest_b_main = _manifest_subset(manifest_b_rows, selected_main_ids)
    manifest_d_main = _manifest_subset(manifest_d_rows, selected_main_ids)
    manifest_b_sanity = _manifest_subset(manifest_b_rows, selected_sanity_ids)
    manifest_d_sanity = _manifest_subset(manifest_d_rows, selected_sanity_ids)

    if manifest_b_main:
        _write_csv(out_dir / "manifest_B_direct_main.csv", manifest_b_main, list(manifest_b_main[0].keys()))
        _write_csv(out_dir / "manifest_slotB_B_direct_main.csv", manifest_b_main, list(manifest_b_main[0].keys()))
    if manifest_d_main:
        _write_csv(out_dir / "manifest_D_visual_only_main.csv", manifest_d_main, list(manifest_d_main[0].keys()))
        _write_csv(out_dir / "manifest_slotA_D_visual_only_main.csv", manifest_d_main, list(manifest_d_main[0].keys()))
    if manifest_b_sanity:
        _write_csv(out_dir / "manifest_B_direct_sanity.csv", manifest_b_sanity, list(manifest_b_sanity[0].keys()))
        _write_csv(out_dir / "manifest_slotB_B_direct_sanity.csv", manifest_b_sanity, list(manifest_b_sanity[0].keys()))
    if manifest_d_sanity:
        _write_csv(out_dir / "manifest_D_visual_only_sanity.csv", manifest_d_sanity, list(manifest_d_sanity[0].keys()))
        _write_csv(out_dir / "manifest_slotA_D_visual_only_sanity.csv", manifest_d_sanity, list(manifest_d_sanity[0].keys()))

    summary_lines = [
        "# B-vs-D Visual-Only Mechanism Pack",
        "",
        f"- total merged rows: `{len(merged_rows)}`",
        f"- excluded existing pack10 ids: `{len(excluded_ids)}`",
        f"- scored candidates: `{len(candidate_rows)}`",
        f"- eligible candidates: `{len(eligible_rows)}`",
        f"- selected main size: `{len(selected_main)}`",
        f"- selected sanity size: `{len(selected_sanity)}`",
        "",
        "## Main pack by image dependence group",
        "",
    ]

    dep_counts: dict[str, int] = defaultdict(int)
    op_counts: dict[str, int] = defaultdict(int)
    disagree_count = 0
    for row in selected_main:
        dep_counts[row["image_dependence_group"]] += 1
        op_counts[row["reasoning_operation"]] += 1
        disagree_count += int(row["has_behavior_disagreement"])

    for dep in sorted(dep_counts):
        summary_lines.append(f"- `{dep}`: `{dep_counts[dep]}`")

    summary_lines.extend(
        [
            "",
            "## Main pack by reasoning operation",
            "",
        ]
    )
    for op in sorted(op_counts):
        summary_lines.append(f"- `{op}`: `{op_counts[op]}`")

    summary_lines.extend(
        [
            "",
            f"- behavior disagreement rows in main pack: `{disagree_count}`",
            "",
            "## Sanity subset",
            "",
        ]
    )
    for row in selected_sanity:
        summary_lines.append(
            f"- `{row['sample_id']}` | `{row['reasoning_operation']}` | "
            f"`{row['image_dependence_group']}` | "
            f"`delta_vqa={row['delta_vqa_score']}` | "
            f"`B={row['B_strict']}` | `D={row['D_strict']}` | "
            f"{row['question_text']}"
        )

    (out_dir / "selection_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"[done] selected_main={len(selected_main)}")
    print(f"[done] selected_sanity={len(selected_sanity)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
