#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(r"E:\Bridging")
PAPERPACK = ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
CROSS = ROOT / "doc" / "experiments" / "stage3" / "cross_model"

PROMPT_TEMPLATES = {
    "B_direct": "{question} Reply with only one short sentence in exactly this format: The answer is <short answer>.",
    "D_visual_only": "{question} Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_path(pack: str) -> Path:
    if pack == "primary":
        return PAPERPACK / "paperpack72_primary_prompt_runs.csv"
    if pack == "strict":
        return PAPERPACK / "paperpack72_strict_sensitivity_prompt_runs.csv"
    raise ValueError(f"unsupported pack: {pack}")


def _image_id_from_filename(image_filename: str) -> str:
    match = re.search(r"(\d{12})", image_filename or "")
    if not match:
        return ""
    return str(int(match.group(1)))


def _question_id_from_sample_id(sample_id: str) -> str:
    match = re.search(r"(\d+)$", sample_id or "")
    return match.group(1) if match else ""


def _gemma_eval_row(row: dict[str, str], pack: str) -> dict[str, str]:
    prompt_name = row["prompt_name"].strip()
    question_text = row["question_text"].strip()
    answer_text = row["answer_text"].strip()
    image_filename = Path(row["image_filename"]).name
    image_id = _image_id_from_filename(image_filename)
    question_id = _question_id_from_sample_id(row["sample_id"])
    question = PROMPT_TEMPLATES[prompt_name].format(question=question_text)
    notes = ";".join(
        [
            f"answer={answer_text}",
            f"qid={question_id}",
            f"image_id={image_id}",
            f"pack={pack}",
            f"prompt_name={prompt_name}",
        ]
    )
    return {
        "sample_id": row["sample_id"].strip(),
        "question_id": question_id,
        "image_id": image_id,
        "image_filename": image_filename,
        "image_path": row["local_image_path"].strip(),
        "question": question,
        "question_text": question_text,
        "gold_answer": answer_text,
        "answer_text": answer_text,
        "prompt_name": prompt_name,
        "pack": pack,
        "reasoning_operation": row.get("reasoning_operation", ""),
        "image_dependence_tier": row.get("image_dependence_tier", ""),
        "paperpack_source": row.get("paperpack_source", ""),
        "mask_dir": row.get("mask_dir", ""),
        "notes": notes,
    }


def _choose_smoke_samples(primary_rows: list[dict[str, str]], count: int) -> list[str]:
    by_sample: dict[str, dict[str, str]] = {}
    for row in primary_rows:
        by_sample.setdefault(row["sample_id"], row)
    candidates = [
        row
        for row in by_sample.values()
        if row.get("image_dependence_tier") == "strong"
        and row.get("paperpack_source") == "original"
        and row.get("reasoning_operation") in {"visual_readout", "symbol_text_reading", "compact_scene_inference"}
    ]
    candidates.sort(
        key=lambda row: (
            {"visual_readout": 0, "symbol_text_reading": 1, "compact_scene_inference": 2}.get(
                row.get("reasoning_operation"), 9
            ),
            row.get("sample_id", ""),
        )
    )
    return [row["sample_id"] for row in candidates[:count]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage3 Gemma paperpack eval manifests.")
    parser.add_argument("--smoke-count", type=int, default=3)
    parser.add_argument("--out-dir", default=str(CROSS))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "question_id",
        "image_id",
        "image_filename",
        "image_path",
        "question",
        "question_text",
        "gold_answer",
        "answer_text",
        "prompt_name",
        "pack",
        "reasoning_operation",
        "image_dependence_tier",
        "paperpack_source",
        "mask_dir",
        "notes",
    ]

    report: dict[str, Any] = {"packs": {}, "smoke_count": args.smoke_count}
    primary_prompt_rows = _read_csv(_manifest_path("primary"))
    smoke_ids = set(_choose_smoke_samples(primary_prompt_rows, args.smoke_count))
    smoke_bucket_rows: list[dict[str, str]] = []

    for pack in ["primary", "strict"]:
        prompt_rows = _read_csv(_manifest_path(pack))
        pack_report: dict[str, Any] = {"prompt_rows": len(prompt_rows), "prompts": {}}
        for prompt_name in ["B_direct", "D_visual_only"]:
            rows = [_gemma_eval_row(row, pack) for row in prompt_rows if row.get("prompt_name") == prompt_name]
            out_path = out_dir / f"stage3_gemma_eval_{pack}_{prompt_name}.csv"
            _write_csv(out_path, rows, fieldnames)
            smoke_rows = [row for row in rows if pack == "primary" and row["sample_id"] in smoke_ids]
            if smoke_rows:
                _write_csv(out_dir / f"stage3_gemma_eval_smoke_{prompt_name}.csv", smoke_rows, fieldnames)
            pack_report["prompts"][prompt_name] = {
                "rows": len(rows),
                "unique_samples": len({row["sample_id"] for row in rows}),
                "missing_images": sum(1 for row in rows if not Path(row["image_path"]).exists()),
                "missing_gold_answers": sum(1 for row in rows if not row["gold_answer"]),
                "out_csv": str(out_path),
            }
        report["packs"][pack] = pack_report

    sample_lookup = {row["sample_id"]: row for row in primary_prompt_rows if row["sample_id"] in smoke_ids}
    for sample_id in sorted(smoke_ids):
        row = sample_lookup[sample_id]
        smoke_bucket_rows.append(
            {
                "sample_id": sample_id,
                "bucket": "paperpack_smoke",
                "question_text": row.get("question_text", ""),
                "answer_text": row.get("answer_text", ""),
                "reasoning_operation": row.get("reasoning_operation", ""),
                "image_dependence_tier": row.get("image_dependence_tier", ""),
                "paperpack_source": row.get("paperpack_source", ""),
            }
        )
    _write_csv(
        out_dir / "stage3_gemma_source_tracing_smoke_manifest.csv",
        smoke_bucket_rows,
        [
            "sample_id",
            "bucket",
            "question_text",
            "answer_text",
            "reasoning_operation",
            "image_dependence_tier",
            "paperpack_source",
        ],
    )
    report["smoke_sample_ids"] = sorted(smoke_ids)
    report["smoke_manifest"] = str(out_dir / "stage3_gemma_source_tracing_smoke_manifest.csv")
    _write_json(out_dir / "stage3_gemma_eval_manifest_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
