#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
STAGE3 = ROOT / "doc" / "experiments" / "stage3"
PAPERPACK = STAGE3 / "paperpack72"
CROSS = STAGE3 / "cross_model"
REMOTE_SYNC = ROOT / "remote_sync"

PRIMARY_PROMPT_RUNS = PAPERPACK / "paperpack72_primary_prompt_runs.csv"
STRICT_PROMPT_RUNS = PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"

OUT_CSV = CROSS / "stage3_gemma_paperpack_overlap_report.csv"
OUT_JSON = CROSS / "stage3_gemma_paperpack_overlap_report.json"
OUT_RAW_COMPARE = CROSS / "stage3_gemma_paperpack_overlap_raw_compare_rows.csv"
OUT_RAW_NODES = CROSS / "stage3_gemma_paperpack_overlap_raw_node_rows.csv"
OUT_RAW_INTERVENTION = CROSS / "stage3_gemma_paperpack_overlap_raw_intervention_rows.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_prompt_pack(path: Path, pack: str) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_sample: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            continue
        by_sample.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                f"in_{pack}": "1",
                f"{pack}_prompt_runs": "0",
                "question_text": row.get("question_text", ""),
                "answer_text": row.get("answer_text", ""),
                "reasoning_operation": row.get("reasoning_operation", ""),
                "image_dependence_tier": row.get("image_dependence_tier", ""),
                "paperpack_source": row.get("paperpack_source", ""),
                "image_filename": row.get("image_filename", ""),
            },
        )
        by_sample[sample_id][f"{pack}_prompt_runs"] = str(int(by_sample[sample_id][f"{pack}_prompt_runs"]) + 1)
    return by_sample


def find_existing(pattern: str) -> list[Path]:
    if not REMOTE_SYNC.exists():
        return []
    try:
        return sorted(path for path in REMOTE_SYNC.rglob(pattern) if path.is_file())
    except OSError:
        # Some old remote_sync trees contain stale symlinks. Skip those rather than failing the audit.
        return sorted(path for path in REMOTE_SYNC.glob(f"**/{pattern}") if path.is_file())


def read_rows_from(paths: list[Path], sample_ids: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        try:
            rows = read_csv(path)
        except Exception:
            continue
        for row in rows:
            sample_id = row.get("sample_id", "")
            if sample_id not in sample_ids:
                continue
            record: dict[str, Any] = {"source_file": str(path)}
            record.update(row)
            out.append(record)
    return out


def file_summary(rows: list[dict[str, Any]]) -> str:
    files = sorted({str(row.get("source_file", "")) for row in rows if row.get("source_file")})
    return " | ".join(files)


def unique_nonempty(rows: list[dict[str, Any]], field: str) -> list[str]:
    return sorted({str(row.get(field, "")) for row in rows if row.get(field, "") != ""})


def main() -> int:
    primary = load_prompt_pack(PRIMARY_PROMPT_RUNS, "primary")
    strict = load_prompt_pack(STRICT_PROMPT_RUNS, "strict")
    all_ids = sorted(set(primary) | set(strict))

    compare_paths = find_existing("sample_compare_controlled.csv")
    compare_paths.extend(find_existing("compare_sample_compare_controlled.csv"))
    node_paths = find_existing("nodes_detailed_controlled.csv")
    node_paths.extend(find_existing("compare_nodes_detailed_controlled.csv"))
    intervention_paths = find_existing("intervention_smoke*.csv")

    compare_rows = read_rows_from(compare_paths, set(all_ids))
    node_rows = read_rows_from(node_paths, set(all_ids))
    intervention_rows = read_rows_from(intervention_paths, set(all_ids))

    compare_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    node_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    intervention_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in compare_rows:
        compare_by_sample[str(row.get("sample_id", ""))].append(row)
    for row in node_rows:
        node_by_sample[str(row.get("sample_id", ""))].append(row)
    for row in intervention_rows:
        intervention_by_sample[str(row.get("sample_id", ""))].append(row)

    report_rows: list[dict[str, Any]] = []
    for sample_id in all_ids:
        base: dict[str, Any] = {
            "sample_id": sample_id,
            "in_primary": "1" if sample_id in primary else "0",
            "in_strict": "1" if sample_id in strict else "0",
            "primary_prompt_runs": primary.get(sample_id, {}).get("primary_prompt_runs", "0"),
            "strict_prompt_runs": strict.get(sample_id, {}).get("strict_prompt_runs", "0"),
        }
        source = primary.get(sample_id) or strict.get(sample_id) or {}
        for key in [
            "question_text",
            "answer_text",
            "reasoning_operation",
            "image_dependence_tier",
            "paperpack_source",
            "image_filename",
        ]:
            base[key] = source.get(key, "")

        sample_compare = compare_by_sample.get(sample_id, [])
        sample_nodes = node_by_sample.get(sample_id, [])
        sample_intervention = intervention_by_sample.get(sample_id, [])
        compare_buckets = unique_nonempty(sample_compare, "bucket")
        node_buckets = unique_nonempty(sample_nodes, "bucket")
        intervention_files = unique_nonempty(sample_intervention, "source_file")

        if sample_compare and sample_nodes and sample_intervention:
            status = "historical_compare_nodes_intervention_overlap"
        elif sample_compare and sample_nodes:
            status = "historical_compare_nodes_overlap"
        elif sample_compare or sample_nodes:
            status = "partial_historical_overlap"
        else:
            status = "no_historical_gemma_overlap"

        base.update(
            {
                "historical_overlap_status": status,
                "compare_rows": len(sample_compare),
                "node_rows": len(sample_nodes),
                "intervention_rows": len(sample_intervention),
                "compare_buckets": ",".join(compare_buckets),
                "node_buckets": ",".join(node_buckets),
                "compare_source_files": file_summary(sample_compare),
                "node_source_files": file_summary(sample_nodes),
                "intervention_source_files": " | ".join(intervention_files),
            }
        )
        report_rows.append(base)

    fieldnames = [
        "sample_id",
        "in_primary",
        "in_strict",
        "primary_prompt_runs",
        "strict_prompt_runs",
        "question_text",
        "answer_text",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
        "image_filename",
        "historical_overlap_status",
        "compare_rows",
        "node_rows",
        "intervention_rows",
        "compare_buckets",
        "node_buckets",
        "compare_source_files",
        "node_source_files",
        "intervention_source_files",
    ]
    write_csv(OUT_CSV, report_rows, fieldnames)
    write_csv(OUT_RAW_COMPARE, compare_rows, list(compare_rows[0].keys()) if compare_rows else ["source_file"])
    write_csv(OUT_RAW_NODES, node_rows, list(node_rows[0].keys()) if node_rows else ["source_file"])
    write_csv(
        OUT_RAW_INTERVENTION,
        intervention_rows,
        list(intervention_rows[0].keys()) if intervention_rows else ["source_file"],
    )

    available_samples = [
        row["sample_id"]
        for row in report_rows
        if row["historical_overlap_status"] in {"historical_compare_nodes_intervention_overlap", "historical_compare_nodes_overlap"}
    ]
    available_prompt_runs = sum(
        int(row["primary_prompt_runs"])
        for row in report_rows
        if row["sample_id"] in set(available_samples) and row["in_primary"] == "1"
    )
    status_counts: dict[str, int] = defaultdict(int)
    for row in report_rows:
        status_counts[str(row["historical_overlap_status"])] += 1

    decision_status = (
        "historical_confirmatory_overlap_possible"
        if available_prompt_runs >= 20
        else "limited_calibration_only"
    )
    payload = {
        "status": decision_status,
        "interpretation": (
            "This is an overlap audit only. If overlap is below 20 primary prompt-runs, historical Gemma rows "
            "should be treated as limited calibration and cannot replace a Gemma paperpack source-tracing rerun."
        ),
        "primary_samples": len(primary),
        "strict_samples": len(strict),
        "union_samples": len(all_ids),
        "historical_available_samples": len(available_samples),
        "historical_available_primary_prompt_runs": available_prompt_runs,
        "status_counts": dict(sorted(status_counts.items())),
        "scanned_files": {
            "sample_compare_controlled": len(compare_paths),
            "nodes_detailed_controlled": len(node_paths),
            "intervention_smoke": len(intervention_paths),
        },
        "artifacts": {
            "report_csv": str(OUT_CSV),
            "raw_compare_rows": str(OUT_RAW_COMPARE),
            "raw_node_rows": str(OUT_RAW_NODES),
            "raw_intervention_rows": str(OUT_RAW_INTERVENTION),
        },
    }
    write_json(OUT_JSON, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
