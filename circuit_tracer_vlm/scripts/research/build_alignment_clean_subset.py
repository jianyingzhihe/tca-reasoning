#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PROMPT_A_SUFFIX = " Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>."
PROMPT_B_SUFFIX = " Reply exactly in the format: The answer is <short answer>."
CANONICAL_ASSISTANT_PREFIX = "The answer is "


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_compare_dir(run_root: Path) -> Path:
    direct = run_root / "sample_compare_controlled.csv"
    if direct.exists():
        return run_root
    matches = sorted({p.parent for p in run_root.rglob("sample_compare_controlled.csv")})
    if not matches:
        raise FileNotFoundError(f"could not find sample_compare_controlled.csv under {run_root}")
    if len(matches) > 1:
        # Prefer the compare dir whose name matches the run-root leaf if possible.
        for match in matches:
            if match.name == run_root.name:
                return match
    return matches[0]


def _strip_prompt_suffix(question: str, suffix: str) -> tuple[str, bool]:
    if question.endswith(suffix):
        return question[: -len(suffix)], True
    return question, False


def _bool_str(value: bool) -> str:
    return "True" if value else "False"


def _load_optional_sample_meta(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if sample_id:
            out[sample_id] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an alignment-clean Stage 1 subset summary from answer-aligned bucket runs. "
            "This is intended to identify samples suitable for later multimodal follow-up "
            "(wrong-image, region mask, restoration) without prompt/prefix/alignment confusion."
        )
    )
    parser.add_argument(
        "--run-roots",
        nargs="+",
        required=True,
        help="One or more bucket run roots such as ..._A1_B1, ..._A0_B1, etc.",
    )
    parser.add_argument(
        "--sample-meta-csv",
        default="",
        help="Optional CSV keyed by sample_id (e.g. targeted pack manifest) to merge labels/priority info.",
    )
    parser.add_argument(
        "--prompt-a-suffix",
        default=PROMPT_A_SUFFIX,
        help="Expected suffix for run A question text.",
    )
    parser.add_argument(
        "--prompt-b-suffix",
        default=PROMPT_B_SUFFIX,
        help="Expected suffix for run B question text.",
    )
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    sample_meta = _load_optional_sample_meta(
        Path(args.sample_meta_csv).expanduser().resolve() if args.sample_meta_csv else None
    )

    rows_out: list[dict[str, str]] = []
    summary_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for run_root_arg in args.run_roots:
        run_root = Path(run_root_arg).expanduser().resolve()
        bucket = run_root.name.split("_")[-2] + "_" + run_root.name.split("_")[-1]
        compare_dir = _find_compare_dir(run_root)

        meta_a_by_sample = {row["sample_id"]: row for row in _read_csv(run_root / "answer_aligned_meta_a.csv")}
        meta_b_by_sample = {row["sample_id"]: row for row in _read_csv(run_root / "answer_aligned_meta_b.csv")}
        compare_rows = _read_csv(compare_dir / "sample_compare_controlled.csv")
        node_rows = _read_csv(compare_dir / "nodes_detailed_controlled.csv")

        intervention_path = run_root / f"intervention_smoke_{bucket}.csv"
        intervention_rows = _read_csv(intervention_path) if intervention_path.exists() else []

        feature_counts: dict[tuple[str, str], int] = defaultdict(int)
        for node in node_rows:
            if node.get("node_type") != "feature":
                continue
            sample_id = node.get("sample_id", "")
            run = node.get("run", "")
            feature_counts[(sample_id, run)] += 1

        intervention_counts: dict[tuple[str, str], int] = defaultdict(int)
        for row in intervention_rows:
            sample_id = row.get("sample_id", "")
            run = row.get("run", "")
            intervention_counts[(sample_id, run)] += 1

        for compare_row in compare_rows:
            sample_id = compare_row["sample_id"]
            meta_a = meta_a_by_sample[sample_id]
            meta_b = meta_b_by_sample[sample_id]
            merged = sample_meta.get(sample_id, {})

            a_question = meta_a.get("question", "")
            b_question = meta_b.get("question", "")
            a_base_question, a_question_canonical = _strip_prompt_suffix(a_question, args.prompt_a_suffix)
            b_base_question, b_question_canonical = _strip_prompt_suffix(b_question, args.prompt_b_suffix)

            a_assistant_prefix = meta_a.get("assistant_prefix", "")
            b_assistant_prefix = meta_b.get("assistant_prefix", "")
            a_assistant_prefix_canonical = a_assistant_prefix == CANONICAL_ASSISTANT_PREFIX
            b_assistant_prefix_canonical = b_assistant_prefix == CANONICAL_ASSISTANT_PREFIX

            a_target = (meta_a.get("target_token_id") or "").strip()
            b_target = (meta_b.get("target_token_id") or "").strip()
            same_target_nonempty = bool(a_target) and a_target == b_target

            a_has_hint = "hint" in (meta_a.get("generated_text", "") or "").lower() or "hint" in a_assistant_prefix.lower()
            b_has_hint = "hint" in (meta_b.get("generated_text", "") or "").lower() or "hint" in b_assistant_prefix.lower()

            a_status_ok = (meta_a.get("status") or "").strip().lower() == "ok"
            b_status_ok = (meta_b.get("status") or "").strip().lower() == "ok"
            base_question_match = a_base_question == b_base_question

            feature_count_a = feature_counts[(sample_id, "A")]
            feature_count_b = feature_counts[(sample_id, "B")]
            intervention_count_a = intervention_counts[(sample_id, "A")]
            intervention_count_b = intervention_counts[(sample_id, "B")]

            clean_core = (
                same_target_nonempty
                and a_status_ok
                and b_status_ok
                and a_question_canonical
                and b_question_canonical
                and base_question_match
            )
            clean_prompt = clean_core and a_assistant_prefix_canonical and b_assistant_prefix_canonical and not a_has_hint and not b_has_hint
            clean_core_any_run = clean_core and (intervention_count_a > 0 or intervention_count_b > 0)
            clean_core_ab_pair = clean_core and intervention_count_a > 0 and intervention_count_b > 0
            clean_ab_pair = clean_prompt and intervention_count_a > 0 and intervention_count_b > 0
            clean_any_run = clean_prompt and (intervention_count_a > 0 or intervention_count_b > 0)

            out_row = {
                "bucket": bucket,
                "sample_id": sample_id,
                "display_question": a_base_question,
                "answer_text": meta_a.get("answer_text", ""),
                "same_target_nonempty": _bool_str(same_target_nonempty),
                "a_status_ok": _bool_str(a_status_ok),
                "b_status_ok": _bool_str(b_status_ok),
                "a_question_canonical": _bool_str(a_question_canonical),
                "b_question_canonical": _bool_str(b_question_canonical),
                "base_question_match": _bool_str(base_question_match),
                "a_assistant_prefix_canonical": _bool_str(a_assistant_prefix_canonical),
                "b_assistant_prefix_canonical": _bool_str(b_assistant_prefix_canonical),
                "a_has_hint": _bool_str(a_has_hint),
                "b_has_hint": _bool_str(b_has_hint),
                "a_target_token_id": a_target,
                "b_target_token_id": b_target,
                "feature_count_a": str(feature_count_a),
                "feature_count_b": str(feature_count_b),
                "intervention_success_count_a": str(intervention_count_a),
                "intervention_success_count_b": str(intervention_count_b),
                "node_overlap_jaccard": compare_row.get("node_overlap_jaccard", ""),
                "edge_overlap_jaccard": compare_row.get("edge_overlap_jaccard", ""),
                "delta_target_feature_ratio": compare_row.get("delta_target_feature_ratio", ""),
                "delta_target_token_ratio": compare_row.get("delta_target_token_ratio", ""),
                "clean_core": _bool_str(clean_core),
                "clean_prompt": _bool_str(clean_prompt),
                "clean_core_any_run": _bool_str(clean_core_any_run),
                "clean_core_ab_pair": _bool_str(clean_core_ab_pair),
                "clean_any_run": _bool_str(clean_any_run),
                "clean_ab_pair": _bool_str(clean_ab_pair),
                "priority": merged.get("priority", ""),
                "rank": merged.get("rank", ""),
                "visual_type_label": merged.get("visual_type_label", ""),
                "knowledge_level_label": merged.get("knowledge_level_label", ""),
                "question_type": merged.get("question_type", ""),
                "visual_structure": merged.get("visual_structure", ""),
                "image_dependence": merged.get("image_dependence", ""),
                "image_dependence_group": merged.get("image_dependence_group", ""),
                "reasoning_operation": merged.get("reasoning_operation", ""),
                "ambiguity_flag": merged.get("ambiguity_flag", ""),
            }
            rows_out.append(out_row)

            summary_counter[bucket]["total_samples"] += 1
            summary_counter[bucket]["clean_core"] += int(clean_core)
            summary_counter[bucket]["clean_prompt"] += int(clean_prompt)
            summary_counter[bucket]["clean_core_any_run"] += int(clean_core_any_run)
            summary_counter[bucket]["clean_core_ab_pair"] += int(clean_core_ab_pair)
            summary_counter[bucket]["clean_any_run"] += int(clean_any_run)
            summary_counter[bucket]["clean_ab_pair"] += int(clean_ab_pair)
            summary_counter[bucket]["hint_contaminated"] += int(a_has_hint or b_has_hint)
            summary_counter[bucket]["assistant_prefix_noncanonical"] += int(
                (not a_assistant_prefix_canonical) or (not b_assistant_prefix_canonical)
            )

    rows_out.sort(
        key=lambda row: (
            row["bucket"],
            -int(row["clean_ab_pair"] == "True"),
            -int(row["clean_any_run"] == "True"),
            -int(row["clean_prompt"] == "True"),
            row["sample_id"],
        )
    )

    summary_rows: list[dict[str, str]] = []
    total_counter: Counter[str] = Counter()
    for bucket in sorted(summary_counter):
        counter = summary_counter[bucket]
        total_counter.update(counter)
        summary_rows.append(
            {
                "bucket": bucket,
                "total_samples": str(counter["total_samples"]),
                "clean_core": str(counter["clean_core"]),
                "clean_prompt": str(counter["clean_prompt"]),
                "clean_core_any_run": str(counter["clean_core_any_run"]),
                "clean_core_ab_pair": str(counter["clean_core_ab_pair"]),
                "clean_any_run": str(counter["clean_any_run"]),
                "clean_ab_pair": str(counter["clean_ab_pair"]),
                "hint_contaminated": str(counter["hint_contaminated"]),
                "assistant_prefix_noncanonical": str(counter["assistant_prefix_noncanonical"]),
            }
        )
    summary_rows.append(
        {
            "bucket": "__all__",
            "total_samples": str(total_counter["total_samples"]),
            "clean_core": str(total_counter["clean_core"]),
            "clean_prompt": str(total_counter["clean_prompt"]),
            "clean_core_any_run": str(total_counter["clean_core_any_run"]),
            "clean_core_ab_pair": str(total_counter["clean_core_ab_pair"]),
            "clean_any_run": str(total_counter["clean_any_run"]),
            "clean_ab_pair": str(total_counter["clean_ab_pair"]),
            "hint_contaminated": str(total_counter["hint_contaminated"]),
            "assistant_prefix_noncanonical": str(total_counter["assistant_prefix_noncanonical"]),
        }
    )

    fieldnames = [
        "bucket",
        "sample_id",
        "display_question",
        "answer_text",
        "same_target_nonempty",
        "a_status_ok",
        "b_status_ok",
        "a_question_canonical",
        "b_question_canonical",
        "base_question_match",
        "a_assistant_prefix_canonical",
        "b_assistant_prefix_canonical",
        "a_has_hint",
        "b_has_hint",
        "a_target_token_id",
        "b_target_token_id",
        "feature_count_a",
        "feature_count_b",
        "intervention_success_count_a",
        "intervention_success_count_b",
        "node_overlap_jaccard",
        "edge_overlap_jaccard",
        "delta_target_feature_ratio",
        "delta_target_token_ratio",
        "clean_core",
        "clean_prompt",
        "clean_core_any_run",
        "clean_core_ab_pair",
        "clean_any_run",
        "clean_ab_pair",
        "priority",
        "rank",
        "visual_type_label",
        "knowledge_level_label",
        "question_type",
        "visual_structure",
        "image_dependence",
        "image_dependence_group",
        "reasoning_operation",
        "ambiguity_flag",
    ]
    _write_csv(Path(args.out_csv).expanduser().resolve(), rows_out, fieldnames)
    _write_csv(
        Path(args.summary_csv).expanduser().resolve(),
        summary_rows,
        [
            "bucket",
            "total_samples",
            "clean_core",
            "clean_prompt",
            "clean_core_any_run",
            "clean_core_ab_pair",
            "clean_any_run",
            "clean_ab_pair",
            "hint_contaminated",
            "assistant_prefix_noncanonical",
        ],
    )
    print(f"[done] out_csv={Path(args.out_csv).expanduser().resolve()}")
    print(f"[done] summary_csv={Path(args.summary_csv).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
