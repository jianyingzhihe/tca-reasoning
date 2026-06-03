#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def _load_meta(outputs_root: Path, run_tag_base: str, bucket: str, run: str) -> dict[str, dict[str, str]]:
    suffix = "a" if run == "A" else "b"
    path = outputs_root / f"{run_tag_base}_{bucket}" / f"answer_aligned_meta_{suffix}.csv"
    if not path.exists():
        return {}
    return {row.get("sample_id", ""): row for row in _read_csv(path)}


def _node_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row.get("bucket", ""),
        row.get("sample_id", ""),
        row.get("run", ""),
        row.get("node_role", ""),
        row.get("feature_layer", ""),
        row.get("feature_pos", ""),
        row.get("feature_id", ""),
    )


def _build_multimodal_batch(processor, image, prompt: str, assistant_prefix: str = ""):
    prompt = (prompt or "").strip()
    assistant_prefix = assistant_prefix or ""

    if "<start_of_image>" in prompt:
        if assistant_prefix:
            text = f"{prompt.rstrip()} {assistant_prefix.lstrip()}".strip()
        else:
            text = prompt
        return processor(text=text, images=image, return_tensors="pt")

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if assistant_prefix:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_prefix}],
                }
            )
            add_generation_prompt = False
        else:
            add_generation_prompt = True

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if isinstance(inputs, dict) and "input_ids" in inputs:
            return inputs
    except Exception:
        pass

    if assistant_prefix:
        text = f"<start_of_image> {prompt}\n\n{assistant_prefix}".strip()
    else:
        text = f"<start_of_image> {prompt}".strip()
    return processor(text=text, images=image, return_tensors="pt")


def _apply_center_mask(
    image: Image.Image,
    *,
    mask_fraction: float,
    mask_fill_rgb: tuple[int, int, int],
) -> Image.Image:
    masked = image.copy()
    width, height = masked.size
    mask_w = max(1, int(round(width * mask_fraction)))
    mask_h = max(1, int(round(height * mask_fraction)))
    left = max(0, (width - mask_w) // 2)
    top = max(0, (height - mask_h) // 2)
    right = min(width, left + mask_w)
    bottom = min(height, top + mask_h)
    ImageDraw.Draw(masked).rectangle((left, top, right, bottom), fill=mask_fill_rgb)
    return masked


def _condition_image(
    condition: str,
    original_image: Image.Image,
    pilot_row: dict[str, str],
    *,
    default_mask_fraction: float,
    default_mask_fill: tuple[int, int, int],
) -> Image.Image:
    if condition == "clean":
        return original_image
    if condition == "no_image":
        return Image.new("RGB", original_image.size, color=(128, 128, 128))
    if condition == "wrong_image":
        wrong_path = (pilot_row.get("condition_image_path") or "").strip()
        if not wrong_path:
            raise ValueError("missing wrong-image path")
        return Image.open(wrong_path).convert("RGB")
    if condition == "masked_image":
        mask_fraction = _safe_float(pilot_row.get("mask_fraction"))
        if mask_fraction is None:
            mask_fraction = default_mask_fraction
        mask_fill_raw = (pilot_row.get("mask_fill_rgb") or "").strip()
        if mask_fill_raw:
            parts = [int(part.strip()) for part in mask_fill_raw.split(",")]
            mask_fill = (parts[0], parts[1], parts[2])
        else:
            mask_fill = default_mask_fill
        return _apply_center_mask(
            original_image,
            mask_fraction=mask_fraction,
            mask_fill_rgb=mask_fill,
        )
    raise ValueError(f"unknown condition: {condition}")


def _seq_len(batch: dict) -> int:
    input_ids = batch["input_ids"]
    return int(input_ids.shape[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a restoration-ready subset by checking whether traced feature positions are in range for intervention-time batches."
    )
    parser.add_argument("--run-tag-base", required=True)
    parser.add_argument("--pilot-csv", required=True)
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--default-mask-fraction", type=float, default=0.4)
    parser.add_argument("--default-mask-fill", default="128,128,128")
    parser.add_argument("--out-node-csv", required=True)
    parser.add_argument("--out-pilot-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    pilot_rows = _read_csv(Path(args.pilot_csv).expanduser().resolve())
    if not pilot_rows:
        raise ValueError("no pilot rows loaded")

    mask_fill_parts = [int(part.strip()) for part in args.default_mask_fill.split(",")]
    if len(mask_fill_parts) != 3:
        raise ValueError("--default-mask-fill must have exactly 3 comma-separated integers")
    default_mask_fill = (mask_fill_parts[0], mask_fill_parts[1], mask_fill_parts[2])

    rows_by_node: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in pilot_rows:
        rows_by_node[_node_key(row)][row.get("condition", "")] = row

    buckets = sorted({row.get("bucket", "") for row in pilot_rows})
    meta_a_by_bucket = {bucket: _load_meta(outputs_root, args.run_tag_base, bucket, "A") for bucket in buckets}
    meta_b_by_bucket = {bucket: _load_meta(outputs_root, args.run_tag_base, bucket, "B") for bucket in buckets}

    processor = AutoProcessor.from_pretrained(args.model_name)

    node_rows: list[dict[str, str]] = []
    ready_node_keys: set[tuple[str, ...]] = set()
    summary_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for node_key, by_condition in sorted(rows_by_node.items()):
        bucket, sample_id, run, node_role, feature_layer, feature_pos, feature_id = node_key
        meta_map = meta_a_by_bucket[bucket] if run == "A" else meta_b_by_bucket[bucket]
        meta = meta_map.get(sample_id, {})
        question = meta.get("question", by_condition.get("clean", {}).get("question", ""))
        assistant_prefix = meta.get("assistant_prefix", "")
        image_path = (meta.get("image_path") or by_condition.get("clean", {}).get("image_path") or "").strip()
        pos = _safe_int(feature_pos)
        if not image_path or pos is None:
            continue

        original_image = Image.open(image_path).convert("RGB")

        row_out = {
            "bucket": bucket,
            "sample_id": sample_id,
            "run": run,
            "node_role": node_role,
            "feature_layer": feature_layer,
            "feature_pos": feature_pos,
            "feature_id": feature_id,
            "question": question,
            "image_path": image_path,
        }

        condition_ready_flags: list[bool] = []
        for condition in ("clean", "wrong_image", "masked_image", "no_image"):
            pilot_row = by_condition.get(condition)
            if pilot_row is None:
                row_out[f"{condition}_seq_len"] = ""
                row_out[f"{condition}_pos_in_range"] = ""
                continue
            condition_image = _condition_image(
                condition,
                original_image,
                pilot_row,
                default_mask_fraction=args.default_mask_fraction,
                default_mask_fill=default_mask_fill,
            )
            batch = _build_multimodal_batch(
                processor,
                condition_image,
                f"<start_of_image> {question}",
                assistant_prefix=assistant_prefix,
            )
            seq_len = _seq_len(batch)
            in_range = pos < seq_len
            row_out[f"{condition}_seq_len"] = str(seq_len)
            row_out[f"{condition}_pos_in_range"] = "true" if in_range else "false"
            if condition != "clean":
                condition_ready_flags.append(in_range)

        all_corrupt_ready = bool(condition_ready_flags) and all(condition_ready_flags)
        clean_ready = row_out.get("clean_pos_in_range") == "true"
        restoration_ready = clean_ready and all_corrupt_ready
        row_out["restoration_ready"] = "true" if restoration_ready else "false"

        if restoration_ready:
            ready_node_keys.add(node_key)

        node_rows.append(row_out)
        summary_counter[(node_role, row_out["restoration_ready"])]["nodes"] += 1
        summary_counter[(node_role, row_out["restoration_ready"])][bucket] += 1

    filtered_pilot_rows = [row for row in pilot_rows if _node_key(row) in ready_node_keys]

    summary_rows: list[dict[str, str]] = []
    for (node_role, ready_flag), counter in sorted(summary_counter.items()):
        summary_rows.append(
            {
                "node_role": node_role,
                "restoration_ready": ready_flag,
                "node_count": str(counter.get("nodes", 0)),
                "A0_B0": str(counter.get("A0_B0", 0)),
                "A0_B1": str(counter.get("A0_B1", 0)),
                "A1_B0": str(counter.get("A1_B0", 0)),
                "A1_B1": str(counter.get("A1_B1", 0)),
            }
        )

    _write_csv(
        Path(args.out_node_csv).expanduser().resolve(),
        node_rows,
        [
            "bucket",
            "sample_id",
            "run",
            "node_role",
            "feature_layer",
            "feature_pos",
            "feature_id",
            "question",
            "image_path",
            "clean_seq_len",
            "clean_pos_in_range",
            "wrong_image_seq_len",
            "wrong_image_pos_in_range",
            "masked_image_seq_len",
            "masked_image_pos_in_range",
            "no_image_seq_len",
            "no_image_pos_in_range",
            "restoration_ready",
        ],
    )
    _write_csv(
        Path(args.out_pilot_csv).expanduser().resolve(),
        filtered_pilot_rows,
        list(filtered_pilot_rows[0].keys()) if filtered_pilot_rows else list(pilot_rows[0].keys()),
    )
    _write_csv(
        Path(args.out_summary_csv).expanduser().resolve(),
        summary_rows,
        ["node_role", "restoration_ready", "node_count", "A0_B0", "A0_B1", "A1_B0", "A1_B1"],
    )
    print(f"[done] node_rows={len(node_rows)} ready_nodes={len(ready_node_keys)} filtered_pilot_rows={len(filtered_pilot_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
