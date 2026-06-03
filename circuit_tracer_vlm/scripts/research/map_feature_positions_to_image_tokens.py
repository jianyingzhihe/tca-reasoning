#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value: object) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _infer_model_name_from_transcoder_set(repo_id: str) -> str:
    from huggingface_hub import hf_hub_download
    import yaml

    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_name = (cfg or {}).get("model_name", "")
    if not model_name:
        raise ValueError(f"model_name missing in {repo_id}/config.yaml")
    return model_name


def _image_token_ids(tokenizer) -> set[int]:
    candidates = [
        "<start_of_image>",
        "<end_of_image>",
        "<image_soft_token>",
        "<image>",
        "<image_token>",
        "<image_start>",
        "<image_end>",
    ]
    out: set[int] = set()
    for token in candidates:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0 and token_id != getattr(tokenizer, "unk_token_id", None):
            out.add(token_id)
    return out


def _token_window(tokenizer, ids: list[int], pos: int, radius: int = 5) -> str:
    chunks = []
    for idx in range(max(0, pos - radius), min(len(ids), pos + radius + 1)):
        tok = tokenizer.convert_ids_to_tokens([ids[idx]])[0]
        mark = "*" if idx == pos else ""
        chunks.append(f"{mark}{idx}:{tok}{mark}")
    return " | ".join(chunks)


def _maybe_grid(index: int | None, count: int) -> tuple[str, str, str]:
    if index is None or count <= 0:
        return "", "", ""
    side = int(round(math.sqrt(count)))
    if side * side != count:
        return "", "", ""
    return str(side), str(index // side), str(index % side)


def main() -> int:
    parser = argparse.ArgumentParser(description="Map traced feature positions to multimodal token/image-token spans.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--transcoder-set", default="tianhux2/gemma3-4b-it-plt")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--pair-ids", default="")
    parser.add_argument("--node-role", default="support")
    args = parser.parse_args()

    from transformers import AutoProcessor

    model_name = args.model.strip() or _infer_model_name_from_transcoder_set(args.transcoder_set)
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    image_ids = _image_token_ids(tokenizer)

    rows = _read_csv(Path(args.manifest_csv).expanduser().resolve())
    sample_ids = {x.strip() for x in args.sample_ids.split(",") if x.strip()}
    pair_ids = {x.strip() for x in args.pair_ids.split(",") if x.strip()}
    rows = [row for row in rows if (row.get("node_role") or "") == args.node_role]
    if sample_ids:
        rows = [row for row in rows if row.get("sample_id") in sample_ids]
    if pair_ids:
        rows = [row for row in rows if row.get("pair_id") in pair_ids]

    out_rows: list[dict[str, str]] = []
    for row in rows:
        image_path = (row.get("image_path") or "").strip()
        image = Image.open(image_path).convert("RGB")
        question = row.get("question", "")
        assistant_prefix = row.get("assistant_prefix", "")
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            if assistant_prefix:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_prefix}]})
            batch = processor.apply_chat_template(
                messages,
                add_generation_prompt=not bool(assistant_prefix),
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            prompt = f"<start_of_image> {question}".strip()
            if assistant_prefix:
                prompt = f"{prompt} {assistant_prefix.lstrip()}".strip()
            batch = processor(text=prompt, images=image, return_tensors="pt")

        ids = [int(x) for x in batch["input_ids"][0].tolist()]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        image_positions = [
            idx
            for idx, (token_id, token) in enumerate(zip(ids, tokens))
            if token_id in image_ids or "image" in str(token).lower()
        ]
        image_pos_to_idx = {pos: idx for idx, pos in enumerate(image_positions)}
        pos = _safe_int(row.get("feature_pos"))
        token_id = ids[pos] if pos is not None and 0 <= pos < len(ids) else None
        token = tokenizer.convert_ids_to_tokens([token_id])[0] if token_id is not None else ""
        image_index = image_pos_to_idx.get(pos) if pos is not None else None
        grid_side, grid_row, grid_col = _maybe_grid(image_index, len(image_positions))
        out_rows.append(
            {
                "pair_id": row.get("pair_id", ""),
                "sample_id": row.get("sample_id", ""),
                "run": row.get("run", ""),
                "prompt_name": row.get("prompt_name", ""),
                "node_role": row.get("node_role", ""),
                "node_source": row.get("node_source", ""),
                "feature_layer": row.get("feature_layer", ""),
                "feature_pos": row.get("feature_pos", ""),
                "feature_id": row.get("feature_id", ""),
                "seq_len": str(len(ids)),
                "token_id_at_pos": "" if token_id is None else str(token_id),
                "token_at_pos": token,
                "image_token_count": str(len(image_positions)),
                "image_span_start": "" if not image_positions else str(min(image_positions)),
                "image_span_end": "" if not image_positions else str(max(image_positions)),
                "is_image_token_pos": str(pos in image_pos_to_idx if pos is not None else False),
                "image_token_index": "" if image_index is None else str(image_index),
                "image_grid_side": grid_side,
                "image_grid_row": grid_row,
                "image_grid_col": grid_col,
                "token_window": _token_window(tokenizer, ids, pos) if pos is not None else "",
            }
        )

    fieldnames = [
        "pair_id",
        "sample_id",
        "run",
        "prompt_name",
        "node_role",
        "node_source",
        "feature_layer",
        "feature_pos",
        "feature_id",
        "seq_len",
        "token_id_at_pos",
        "token_at_pos",
        "image_token_count",
        "image_span_start",
        "image_span_end",
        "is_image_token_pos",
        "image_token_index",
        "image_grid_side",
        "image_grid_row",
        "image_grid_col",
        "token_window",
    ]
    _write_csv(Path(args.out_csv).expanduser().resolve(), out_rows, fieldnames)
    print(f"[done] rows={len(out_rows)} out_csv={args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
