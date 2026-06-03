#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYNC_DIR = ROOT / "remote_sync" / "2026-05-19_stage2a_pretrace_top24"
OUT_DIR = ROOT / "annotation" / "stage2a_region_replication_top24_nearest8"
REMOTE_PACK_DIR = "/root/autodl-tmp/tca-reasoning/annotation/stage2a_region_replication_top24_nearest8"

NEAREST_ENRICHED_CSV = SYNC_DIR / "stage2a_nearest_control_clean_top4_nonzero_enriched.csv"
META_A_CSV = SYNC_DIR / "answer_aligned_meta_a.csv"
META_B_CSV = SYNC_DIR / "answer_aligned_meta_b.csv"

ANNOTATION_PACKS = [
    ROOT / "annotation" / "okvqa_evidence_labelme_round4_core16_extra",
    ROOT / "annotation" / "okvqa_evidence_labelme_round4_ultraeasy16_fresh",
    ROOT / "annotation" / "okvqa_evidence_labelme_round4_core24_easy",
]


RUN_TO_PROMPT = {
    "A": "D_visual_only",
    "B": "B_direct",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)


def _json_label_counts(json_path: Path) -> tuple[int, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    answer = 0
    relate = 0
    for shape in data.get("shapes", []):
        label = (shape.get("label") or "").strip()
        if label == "answer":
            answer += 1
        elif label == "relate":
            relate += 1
    return answer, relate


def _load_meta() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for run, path in (("A", META_A_CSV), ("B", META_B_CSV)):
        for row in _read_csv(path):
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id:
                out[(sample_id, run)] = row
    return out


def _load_annotation_assets() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for pack in ANNOTATION_PACKS:
        manifest = pack / "manifest.csv"
        if not manifest.exists():
            continue
        for row in _read_csv(manifest):
            sample_id = (row.get("sample_id") or "").strip()
            image_filename = (row.get("image_filename") or "").strip()
            if not sample_id or not image_filename:
                continue
            stem = Path(image_filename).stem
            json_path = pack / "images" / f"{stem}.json"
            image_path = pack / "images" / image_filename
            if not json_path.exists() or not image_path.exists():
                continue
            answer_count, relate_count = _json_label_counts(json_path)
            if answer_count <= 0 or relate_count <= 0:
                continue
            if sample_id not in out:
                out[sample_id] = {
                    "annotation_pack": pack.name,
                    "source_image_path": str(image_path),
                    "source_json_path": str(json_path),
                    "image_filename": image_filename,
                    "answer_shape_count": str(answer_count),
                    "relate_shape_count": str(relate_count),
                    "question_text": row.get("question_text", ""),
                    "answer_text": row.get("answer_text", ""),
                    "remote_data_image_path": row.get("remote_image_path", ""),
                }
    return out


def _node_common(
    *,
    pair_id: str,
    nearest_row: dict[str, str],
    meta_row: dict[str, str],
    asset: dict[str, str],
    node_source: str,
    feature_layer: str,
    feature_pos: str,
    feature_id: str,
    reference_delta: str,
) -> dict[str, str]:
    sample_id = nearest_row["sample_id"]
    run = nearest_row["run"]
    image_filename = asset["image_filename"]
    local_image_path = OUT_DIR / "images" / image_filename
    remote_image_path = f"{REMOTE_PACK_DIR}/images/{image_filename}"
    return {
        "pair_id": pair_id,
        "sample_id": sample_id,
        "run": run,
        "prompt_name": RUN_TO_PROMPT.get(run, run),
        "node_role": nearest_row["node_role"],
        "node_source": node_source,
        "assistant_prefix": meta_row.get("assistant_prefix", ""),
        "question": meta_row.get("question", nearest_row.get("question", "")),
        "image_path": str(local_image_path),
        "remote_image_path": remote_image_path,
        "feature_layer": feature_layer,
        "feature_pos": feature_pos,
        "feature_id": feature_id,
        "target_token_id": nearest_row.get("target_token_id", meta_row.get("target_token_id", "")),
        "reasoning_operation": nearest_row.get("reasoning_operation", ""),
        "visual_structure": "",
        "image_dependence": "strong",
        "selection_tier": "stage2a_nearest8",
        "annotation_pack": asset["annotation_pack"],
        "annotation_json_path": str(OUT_DIR / "images" / f"{Path(image_filename).stem}.json"),
        "answer_shape_count": asset["answer_shape_count"],
        "relate_shape_count": asset["relate_shape_count"],
        "reference_clean_delta_target_logit": reference_delta,
        "source_bucket": nearest_row.get("bucket", ""),
        "match_mode": nearest_row.get("match_mode", ""),
        "sampled_match_label": nearest_row.get("sampled_match_label", ""),
        "control_draw_idx": nearest_row.get("control_draw_idx", ""),
        "control_match_label": nearest_row.get("control_match_label", ""),
        "source_feature_layer": nearest_row.get("source_feature_layer", ""),
        "source_feature_pos": nearest_row.get("source_feature_pos", ""),
        "source_feature_id": nearest_row.get("source_feature_id", ""),
        "control_feature_layer": nearest_row.get("control_feature_layer", ""),
        "control_feature_pos": nearest_row.get("control_feature_pos", ""),
        "control_feature_id": nearest_row.get("control_feature_id", ""),
    }


def main() -> int:
    nearest_rows = _read_csv(NEAREST_ENRICHED_CSV)
    meta = _load_meta()
    assets = _load_annotation_assets()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    selected_nearest_rows: list[dict[str, str]] = []
    missing_assets: list[str] = []
    for row in nearest_rows:
        if (row.get("existing_answer_relate_mask") or "").strip().lower() != "true":
            continue
        if row.get("node_role") not in {"support", "suppressor"}:
            continue
        sample_id = row.get("sample_id", "")
        if sample_id not in assets:
            missing_assets.append(sample_id)
            continue
        selected_nearest_rows.append(row)

    if missing_assets:
        unique = sorted(set(missing_assets))
        raise FileNotFoundError(f"missing annotation assets for sample ids: {unique}")
    if not selected_nearest_rows:
        raise ValueError("no selected nearest rows")

    for sample_id in sorted({row["sample_id"] for row in selected_nearest_rows}):
        asset = assets[sample_id]
        image_filename = asset["image_filename"]
        _copy_if_needed(Path(asset["source_image_path"]), OUT_DIR / "images" / image_filename)
        _copy_if_needed(Path(asset["source_json_path"]), OUT_DIR / "images" / f"{Path(image_filename).stem}.json")

    manifest_rows: list[dict[str, str]] = []
    manifest_remote_rows: list[dict[str, str]] = []
    for idx, row in enumerate(selected_nearest_rows, start=1):
        sample_id = row["sample_id"]
        run = row["run"]
        meta_row = meta[(sample_id, run)]
        asset = assets[sample_id]
        pair_id = f"{idx:03d}_{sample_id}_{run}_{row['node_role']}_L{row['source_feature_layer']}_P{row['source_feature_pos']}_F{row['source_feature_id']}"
        source = _node_common(
            pair_id=pair_id,
            nearest_row=row,
            meta_row=meta_row,
            asset=asset,
            node_source="source",
            feature_layer=row.get("source_feature_layer", ""),
            feature_pos=row.get("source_feature_pos", ""),
            feature_id=row.get("source_feature_id", ""),
            reference_delta=row.get("source_delta", ""),
        )
        control = _node_common(
            pair_id=pair_id,
            nearest_row=row,
            meta_row=meta_row,
            asset=asset,
            node_source="nearest_control",
            feature_layer=row.get("control_feature_layer", ""),
            feature_pos=row.get("control_feature_pos", ""),
            feature_id=row.get("control_feature_id", ""),
            reference_delta=row.get("control_delta", ""),
        )
        manifest_rows.extend([source, control])
        for local_row in (source, control):
            remote_row = dict(local_row)
            remote_row["image_path"] = local_row["remote_image_path"]
            remote_row["annotation_json_path"] = f"{REMOTE_PACK_DIR}/images/{Path(local_row['annotation_json_path']).name}"
            manifest_remote_rows.append(remote_row)

    fieldnames = list(manifest_rows[0].keys())
    _write_csv(OUT_DIR / "region_experiment_manifest.csv", manifest_rows, fieldnames)
    _write_csv(OUT_DIR / "region_experiment_manifest_remote.csv", manifest_remote_rows, fieldnames)

    sample_counter = Counter(row["sample_id"] for row in selected_nearest_rows if row["node_role"] == "support")
    role_counter = Counter(row["node_role"] for row in selected_nearest_rows)
    summary = {
        "pack": OUT_DIR.name,
        "selected_pair_rows": len(selected_nearest_rows),
        "manifest_rows": len(manifest_rows),
        "sample_count": len({row["sample_id"] for row in selected_nearest_rows}),
        "support_sample_count": len(sample_counter),
        "role_pair_counts": dict(role_counter),
        "support_pairs_by_sample": dict(sorted(sample_counter.items())),
        "remote_pack_dir": REMOTE_PACK_DIR,
    }
    (OUT_DIR / "selection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Stage 2A Region Replication Pack: top24 nearest8",
        "",
        f"- selected source-control pairs: `{len(selected_nearest_rows)}`",
        f"- manifest rows, source + nearest_control: `{len(manifest_rows)}`",
        f"- sample count: `{summary['sample_count']}`",
        f"- support sample count: `{summary['support_sample_count']}`",
        f"- remote pack dir: `{REMOTE_PACK_DIR}`",
        "",
        "## Role pair counts",
        "",
    ]
    for role, count in sorted(role_counter.items()):
        md.append(f"- `{role}`: `{count}` pairs")
    md.extend(["", "## Support pairs by sample", ""])
    for sample_id, count in sorted(sample_counter.items()):
        asset = assets[sample_id]
        md.append(f"- `{sample_id}`: `{count}` support pairs; annotation pack `{asset['annotation_pack']}`")
    (OUT_DIR / "selection_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[done] out_dir={OUT_DIR}")
    print(f"[done] selected_pair_rows={len(selected_nearest_rows)}")
    print(f"[done] manifest_rows={len(manifest_rows)}")
    print(f"[done] support_sample_count={summary['support_sample_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
