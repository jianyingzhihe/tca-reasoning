#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file


VISUAL_OPTIONS = [
    {
        "id": "localized",
        "title": "Localized",
        "shortcut": "1",
        "description": "One fairly tight local region seems sufficient.",
    },
    {
        "id": "multi_region",
        "title": "Multi-Region",
        "shortcut": "2",
        "description": "Two or more separate regions need to be combined.",
    },
    {
        "id": "diffuse_global",
        "title": "Diffuse / Global",
        "shortcut": "3",
        "description": "Evidence is spread across much of the scene.",
    },
]

KNOWLEDGE_OPTIONS = [
    {
        "id": "low",
        "title": "Low",
        "shortcut": "q",
        "description": "Mostly answerable from the image itself.",
    },
    {
        "id": "medium",
        "title": "Medium",
        "shortcut": "w",
        "description": "Image gives the main clue, but some commonsense helps.",
    },
    {
        "id": "high",
        "title": "High",
        "shortcut": "e",
        "description": "Answer depends heavily on external/world knowledge.",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def create_app(dataset_dir: Path) -> Flask:
    template_dir = Path(__file__).with_name("type_label_ui_templates")
    app = Flask(__name__, template_folder=str(template_dir))

    manifest_path = dataset_dir / "manifest.csv"
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"

    manifest_rows = _read_csv(manifest_path)
    item_map = {row["item_id"]: row for row in manifest_rows}

    label_fieldnames = [
        "item_id",
        "visual_type_label",
        "knowledge_level_label",
        "label_notes",
        "updated_at",
    ]

    def load_labels() -> dict[str, dict[str, str]]:
        if not labels_path.exists():
            return {}
        rows = _read_csv(labels_path)
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            item_id = row.get("item_id")
            if not item_id:
                continue
            if "visual_type_label" not in row and "type_label" in row:
                # Backward compatibility with the earlier single-label prototype.
                row["visual_type_label"] = row.get("type_label", "")
            row.setdefault("visual_type_label", "")
            row.setdefault("knowledge_level_label", "")
            row.setdefault("label_notes", "")
            out[item_id] = row
        return out

    def save_label(item_id: str, visual_type_label: str, knowledge_level_label: str, label_notes: str) -> None:
        labels = load_labels()
        labels[item_id] = {
            "item_id": item_id,
            "visual_type_label": visual_type_label,
            "knowledge_level_label": knowledge_level_label,
            "label_notes": label_notes,
            "updated_at": _now_iso(),
        }
        rows = [labels[key] for key in sorted(labels.keys())]
        _write_csv(labels_path, rows, label_fieldnames)

    def merge_items() -> list[dict[str, str]]:
        labels = load_labels()
        merged = []
        for row in manifest_rows:
            current = dict(row)
            current.update(labels.get(row["item_id"], {}))
            current.setdefault("visual_type_label", "")
            current.setdefault("knowledge_level_label", "")
            current.setdefault("label_notes", "")
            current["image_url"] = f"/image/{row['item_id']}"
            merged.append(current)
        return merged

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            dataset_name=dataset_dir.name,
            visual_options=VISUAL_OPTIONS,
            knowledge_options=KNOWLEDGE_OPTIONS,
        )

    @app.get("/api/items")
    def api_items():
        return jsonify(
            {
                "dataset_name": dataset_dir.name,
                "items": merge_items(),
                "visual_options": VISUAL_OPTIONS,
                "knowledge_options": KNOWLEDGE_OPTIONS,
            }
        )

    @app.post("/api/label")
    def api_label():
        payload = request.get_json(force=True)
        item_id = (payload.get("item_id") or "").strip()
        visual_type_label = (payload.get("visual_type_label") or "").strip()
        knowledge_level_label = (payload.get("knowledge_level_label") or "").strip()
        label_notes = (payload.get("label_notes") or "").strip()
        if item_id not in item_map:
            abort(404, f"unknown item_id: {item_id}")
        valid_visual = {opt["id"] for opt in VISUAL_OPTIONS} | {""}
        valid_knowledge = {opt["id"] for opt in KNOWLEDGE_OPTIONS} | {""}
        if visual_type_label not in valid_visual:
            abort(400, f"invalid visual_type_label: {visual_type_label}")
        if knowledge_level_label not in valid_knowledge:
            abort(400, f"invalid knowledge_level_label: {knowledge_level_label}")
        save_label(item_id, visual_type_label, knowledge_level_label, label_notes)
        return jsonify({"ok": True})

    @app.get("/image/<item_id>")
    def image(item_id: str):
        row = item_map.get(item_id)
        if row is None:
            abort(404, f"unknown item_id: {item_id}")
        image_path = images_dir / row["image_filename"]
        if not image_path.exists():
            abort(404, f"missing image: {image_path}")
        return send_file(image_path)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a local UI for OKVQA sample-type labeling.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    app = create_app(dataset_dir)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
