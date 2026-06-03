#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or "")
    except Exception:
        return default


def _read_round2_sample_ids(round2_manifest: Path) -> set[str]:
    return {
        (row.get("sample_id") or "").strip()
        for row in _read_csv(round2_manifest)
        if (row.get("sample_id") or "").strip()
    }


def _load_meta_maps(cache_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    out: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for path in sorted(cache_dir.glob("meta_*.csv")):
        stem = path.stem  # meta_A1_B0_b
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        bucket = "_".join(parts[1:3])
        run = parts[3].upper()
        rows = _read_csv(path)
        out[(bucket, run)] = {(r.get("sample_id") or "").strip(): r for r in rows}
    return out


def _choose_high_priority_rows(cache_dir: Path, excluded_sample_ids: set[str]) -> list[dict[str, str]]:
    high_rows: list[dict[str, str]] = []
    for name in [
        "intervention_smoke_A0_B1.csv",
        "intervention_smoke_A1_B0.csv",
        "intervention_smoke_A1_B1.csv",
    ]:
        rows = _read_csv(cache_dir / name)
        by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id or sample_id in excluded_sample_ids:
                continue
            by_sample[sample_id].append(row)

        for sample_id, sample_rows in by_sample.items():
            best = max(sample_rows, key=lambda r: abs(_safe_float(r.get("delta_target_logit"))))
            delta = _safe_float(best.get("delta_target_logit"))
            picked = dict(best)
            picked["priority"] = "high"
            picked["source"] = "smoke"
            picked["node_role"] = "support" if delta < 0 else "suppressor"
            high_rows.append(picked)
    return high_rows


def _choose_medium_priority_rows(
    cache_dir: Path,
    excluded_sample_ids: set[str],
) -> list[dict[str, str]]:
    sample_rows = _read_csv(cache_dir / "overnight_candidate_samples.csv")
    feature_rows = _read_csv(cache_dir / "overnight_candidate_features.csv")

    features_by_sample: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in feature_rows:
        bucket = (row.get("bucket") or "").strip()
        sample_id = (row.get("sample_id") or "").strip()
        if not bucket or not sample_id:
            continue
        features_by_sample[(bucket, sample_id)].append(row)

    medium_rows: list[dict[str, str]] = []
    for row in sample_rows:
        bucket = (row.get("bucket") or "").strip()
        if bucket not in {"A0_B1", "A1_B0", "A1_B1"}:
            continue
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id or sample_id in excluded_sample_ids:
            continue
        candidates = features_by_sample.get((bucket, sample_id), [])
        if not candidates:
            continue
        # Prefer top-ranked features, then strongest path mass.
        candidates = sorted(
            candidates,
            key=lambda r: (
                int(r.get("feature_rank_within_run") or "999"),
                -_safe_float(r.get("path_mass_best")),
            ),
        )
        best = dict(candidates[0])
        best["priority"] = "medium"
        best["source"] = "overnight_candidate"
        best["node_role"] = "candidate"
        # Normalize field names to match smoke rows.
        best["question"] = ""
        best["image_path"] = ""
        best["feature_layer"] = best.get("layer", "")
        best["feature_pos"] = best.get("pos", "")
        medium_rows.append(best)
    return medium_rows


def _choose_pending_sample_only_rows(
    cache_dir: Path,
    excluded_sample_ids: set[str],
    meta_maps: dict[tuple[str, str], dict[str, dict[str, str]]],
) -> list[dict[str, str]]:
    sample_rows = _read_csv(cache_dir / "overnight_candidate_samples.csv")
    out: list[dict[str, str]] = []
    for row in sample_rows:
        bucket = (row.get("bucket") or "").strip()
        if bucket not in {"A0_B1", "A1_B0", "A1_B1"}:
            continue
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id or sample_id in excluded_sample_ids:
            continue
        preferred_runs = ["B", "A"] if bucket in {"A0_B1", "A1_B0", "A1_B1"} else ["A", "B"]
        chosen_run = ""
        chosen_meta: dict[str, str] = {}
        for run in preferred_runs:
            meta = meta_maps.get((bucket, run), {}).get(sample_id, {})
            if meta:
                chosen_run = run
                chosen_meta = meta
                break
        if not chosen_meta:
            continue
        out.append(
            {
                "bucket": bucket,
                "sample_id": sample_id,
                "run": chosen_run,
                "priority": "low",
                "source": "candidate_only",
                "node_role": "pending",
                "question": chosen_meta.get("question", ""),
                "image_path": chosen_meta.get("image_path", ""),
                "feature_layer": "",
                "feature_pos": "",
                "feature_id": "",
            }
        )
    return out


def _attach_meta(
    rows: list[dict[str, str]],
    meta_maps: dict[tuple[str, str], dict[str, dict[str, str]]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        bucket = (row.get("bucket") or "").strip()
        run = (row.get("run") or "").strip().upper()
        sample_id = (row.get("sample_id") or "").strip()
        meta = meta_maps.get((bucket, run), {}).get(sample_id, {})
        merged = dict(row)
        if meta:
            merged["question"] = merged.get("question") or meta.get("question", "")
            merged["image_path"] = merged.get("image_path") or meta.get("image_path", "")
            merged["assistant_prefix"] = meta.get("assistant_prefix", "")
            merged["target_token_id"] = meta.get("target_token_id", "")
        out.append(merged)
    return out


def _manifest_row(row: dict[str, str], out_images_dir: Path) -> dict[str, str]:
    remote_image_path = (row.get("image_path") or "").strip()
    image_name = Path(remote_image_path).name
    sample_id = (row.get("sample_id") or "").strip()
    bucket = (row.get("bucket") or "").strip()
    priority = (row.get("priority") or "").strip()
    run = (row.get("run") or "").strip()
    node_role = (row.get("node_role") or "").strip()
    question = (row.get("question") or "").strip()
    return {
        "sample_id": sample_id,
        "bucket": bucket,
        "priority": priority,
        "source": (row.get("source") or "").strip(),
        "run": run,
        "node_role": node_role,
        "question": question,
        "local_image_path": str((out_images_dir / image_name).resolve()),
        "remote_image_path": remote_image_path,
        "feature_layer": (row.get("feature_layer") or "").strip(),
        "feature_pos": (row.get("feature_pos") or "").strip(),
        "feature_id": (row.get("feature_id") or "").strip(),
        "labels_required": "answer,relate",
        "labels_optional": "",
        "annotation_goal": (
            "Mark the most direct answer-bearing visual region as 'answer', "
            "and the broader supporting context as 'relate'."
        ),
    }


def _write_question_sheet(path: Path, manifest_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Round 3 Annotation Sheet",
        "",
        "Use label `answer` for the most direct answer-bearing region.",
        "Use label `relate` for the broader supporting context that still helps answer the question.",
        "",
    ]
    for idx, row in enumerate(manifest_rows, start=1):
        image_name = Path(row["local_image_path"]).name
        lines.extend(
            [
                f"## {idx}. {row['sample_id']} ({row['bucket']}, run {row['run']}, {row['node_role']}, priority {row['priority']}, source {row['source']})",
                "",
                f"- image: `{image_name}`",
                f"- question: {row['question']}",
                "- annotate `answer`: the tight, direct visual evidence for the answer",
                "- annotate `relate`: broader supporting context that still matters for the question",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_quickstart(path: Path, out_dir: Path) -> None:
    text = f"""# OKVQA Evidence Annotation Quickstart (Round 3)

Folder:

- `{out_dir}`

Launcher:

- [launch_labelme.ps1](</{out_dir.as_posix()}/launch_labelme.ps1>)

Question sheet:

- [QUESTION_SHEET.md](</{out_dir.as_posix()}/QUESTION_SHEET.md>)

Manifest:

- [manifest.csv](</{out_dir.as_posix()}/manifest.csv>)
- [manifest.json](</{out_dir.as_posix()}/manifest.json>)

## What to annotate

For each image, please annotate exactly two regions:

1. `answer`
2. `relate`

### `answer`

- the most direct visual evidence for the answer
- if this region were hidden, the question should become much harder

### `relate`

- broader supporting visual context
- still relevant to the question
- may include surrounding objects, scene cues, weather cues, or interaction context

## Important rule

Use the **question text** in the question sheet.

Do not just circle the most salient object.

We care about the region most relevant to the question and answer.
"""
    path.write_text(text, encoding="utf-8")


def _write_launcher(path: Path, images_dir: Path) -> None:
    text = "\n".join(
        [
            f'$imagesDir = "{images_dir}"',
            '$pythonExe = "E:\\code\\conda\\python.exe"',
            "",
            'Write-Host "Launching Labelme on $imagesDir"',
            'Start-Process -FilePath $pythonExe -ArgumentList @("-m", "labelme", $imagesDir)',
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a larger round-3 Labelme package from cached Stage 1 CSVs.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--round2-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    round2_manifest = Path(args.round2_manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    excluded = _read_round2_sample_ids(round2_manifest)
    meta_maps = _load_meta_maps(cache_dir)
    high_rows = _attach_meta(_choose_high_priority_rows(cache_dir, excluded), meta_maps)
    high_ids = {(r.get("sample_id") or "").strip() for r in high_rows}
    medium_rows = _attach_meta(_choose_medium_priority_rows(cache_dir, excluded | high_ids), meta_maps)
    medium_ids = {(r.get("sample_id") or "").strip() for r in medium_rows}
    low_rows = _choose_pending_sample_only_rows(cache_dir, excluded | high_ids | medium_ids, meta_maps)

    selected = high_rows + medium_rows + low_rows
    selected = [r for r in selected if (r.get("question") or "").strip() and (r.get("image_path") or "").strip()]
    selected = sorted(selected, key=lambda r: (r.get("priority") != "high", r.get("bucket", ""), r.get("sample_id", "")))

    manifest_rows = [_manifest_row(row, out_images_dir) for row in selected]
    fieldnames = [
        "sample_id",
        "bucket",
        "priority",
        "source",
        "run",
        "node_role",
        "question",
        "local_image_path",
        "remote_image_path",
        "feature_layer",
        "feature_pos",
        "feature_id",
        "labels_required",
        "labels_optional",
        "annotation_goal",
    ]
    _write_csv(out_dir / "manifest.csv", manifest_rows, fieldnames)
    (out_dir / "manifest.json").write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_question_sheet(out_dir / "QUESTION_SHEET.md", manifest_rows)
    _write_quickstart(out_dir / "ANNOTATION_QUICKSTART.md", out_dir)
    _write_launcher(out_dir / "launch_labelme.ps1", out_images_dir)
    print(f"[done] out_dir={out_dir}")
    print(f"[done] total_rows={len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
