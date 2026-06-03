#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def _clean_question(question: str) -> str:
    text = (question or "").strip()
    suffixes = [
        " Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>.",
        " Reply exactly in the format: The answer is <short answer>.",
    ]
    for suffix in suffixes:
        if text.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


def _priority_from_rank(rank_str: str) -> str:
    try:
        rank = int(rank_str)
    except Exception:
        return "medium"
    if rank <= 4:
        return "high"
    if rank <= 8:
        return "medium"
    return "low"


def _load_meta_maps(cache_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    out: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for path in sorted(cache_dir.glob("meta_*.csv")):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        bucket = "_".join(parts[1:3])
        run = parts[3].upper()
        rows = _read_csv(path)
        out[(bucket, run)] = {(r.get("sample_id") or "").strip(): r for r in rows}
    return out


def _choose_meta_for_sample(
    meta_maps: dict[tuple[str, str], dict[str, dict[str, str]]],
    bucket: str,
    sample_id: str,
) -> tuple[str, dict[str, str]]:
    for run in ("B", "A"):
        row = meta_maps.get((bucket, run), {}).get(sample_id)
        if row:
            return run, row
    return "", {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a sample-typing dataset from cached Stage 1 outputs.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-samples-csv", default="overnight_candidate_samples.csv")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = _read_csv(cache_dir / args.candidate_samples_csv)
    meta_maps = _load_meta_maps(cache_dir)

    manifest_rows: list[dict[str, str]] = []
    for row in candidate_rows:
        bucket = (row.get("bucket") or "").strip()
        sample_id = (row.get("sample_id") or "").strip()
        rank = (row.get("rank") or "").strip()
        chosen_run, meta = _choose_meta_for_sample(meta_maps, bucket, sample_id)
        if not meta:
            continue
        remote_image_path = (meta.get("image_path") or "").strip()
        image_filename = Path(remote_image_path).name
        display_question = _clean_question(meta.get("question", ""))
        answer_text = (meta.get("answer_text") or "").strip()
        manifest_rows.append(
            {
                "item_id": f"{bucket}__{sample_id}",
                "bucket": bucket,
                "rank": rank,
                "priority": _priority_from_rank(rank),
                "sample_id": sample_id,
                "meta_run": chosen_run,
                "display_question": display_question,
                "raw_question": (meta.get("question") or "").strip(),
                "answer_text": answer_text,
                "generated_text": (meta.get("generated_text") or "").strip(),
                "image_filename": image_filename,
                "local_image_path": str((images_dir / image_filename).resolve()),
                "remote_image_path": remote_image_path,
                "same_target_token": (row.get("same_target_token") or "").strip(),
                "intervention_priority_score": (row.get("intervention_priority_score") or "").strip(),
                "filtered_node_overlap_jaccard": (row.get("filtered_node_overlap_jaccard") or "").strip(),
                "delta_target_feature_ratio": (row.get("delta_target_feature_ratio") or "").strip(),
                "delta_target_token_ratio": (row.get("delta_target_token_ratio") or "").strip(),
                "type_label": "",
                "label_notes": "",
            }
        )

    fieldnames = [
        "item_id",
        "bucket",
        "rank",
        "priority",
        "sample_id",
        "meta_run",
        "display_question",
        "raw_question",
        "answer_text",
        "generated_text",
        "image_filename",
        "local_image_path",
        "remote_image_path",
        "same_target_token",
        "intervention_priority_score",
        "filtered_node_overlap_jaccard",
        "delta_target_feature_ratio",
        "delta_target_token_ratio",
        "type_label",
        "label_notes",
    ]
    _write_csv(out_dir / "manifest.csv", manifest_rows, fieldnames)
    (out_dir / "manifest.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] out_dir={out_dir}")
    print(f"[done] total_rows={len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
