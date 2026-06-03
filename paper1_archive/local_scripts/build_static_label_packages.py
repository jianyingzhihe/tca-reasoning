#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(r"E:\Bridging")
SOURCE_DIR = ROOT / "annotation" / "okvqa_type_label_round3_320"
OUTPUT_ROOT = ROOT / "annotation" / "okvqa_type_label_round3_320_static_splits"
N_SPLITS = 4


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
        "shortcut": "Q",
        "description": "Mostly answerable from the image itself.",
    },
    {
        "id": "medium",
        "title": "Medium",
        "shortcut": "W",
        "description": "Image gives the main clue, but some commonsense helps.",
    },
    {
        "id": "high",
        "title": "High",
        "shortcut": "E",
        "description": "Answer depends heavily on external or world knowledge.",
    },
]

LABEL_FIELDS = [
    "item_id",
    "visual_type_label",
    "knowledge_level_label",
    "label_notes",
    "updated_at",
]

README_TEXT = """OKVQA Labeling Package
=====================

This package is fully offline. No Python or local server is required.

How to use
----------
1. Extract the zip file.
2. Open `index.html` in a browser by double-clicking it.
3. For each sample, choose:
   - Visual Evidence Type: Localized / Multi-Region / Diffuse-Global
   - External Knowledge Need: Low / Medium / High
4. Optional: write notes.
5. When finished, click `Export CSV`.
6. Send the exported CSV file back.

Shortcuts
---------
- Visual type: 1 / 2 / 3
- Knowledge level: Q / W / E
- Navigation: Left / Right arrow

Files in this package
---------------------
- `index.html`: labeling UI
- `images/`: sample images
- `manifest.csv`: sample metadata
- `labels.csv`: empty template
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_round_robin(rows: list[dict[str, str]], n_splits: int) -> list[list[dict[str, str]]]:
    priority_order = ["high", "medium", "low"]
    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in priority_order}

    for row in rows:
        priority = (row.get("priority") or "low").strip()
        if priority not in grouped:
            priority = "low"
        grouped[priority].append(row)

    splits: list[list[dict[str, str]]] = [[] for _ in range(n_splits)]
    for priority in priority_order:
        for idx, row in enumerate(grouped[priority]):
            splits[idx % n_splits].append(row)
    return splits


def make_html(dataset_name: str, items: list[dict[str, str]]) -> str:
    payload = {
        "dataset_name": dataset_name,
        "items": items,
        "visual_options": VISUAL_OPTIONS,
        "knowledge_options": KNOWLEDGE_OPTIONS,
    }
    title = html.escape(dataset_name)
    data_json = json.dumps(payload, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f6f8;
      color: #111827;
    }}
    .app {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .title {{
      font-size: 22px;
      font-weight: 700;
    }}
    .sub {{
      color: #4b5563;
      font-size: 14px;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid #d1d5db;
      background: white;
      padding: 10px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
    }}
    button.primary {{
      background: #111827;
      color: white;
      border-color: #111827;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
    }}
    .panel {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 16px;
    }}
    .image-wrap {{
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 520px;
      background: #f9fafb;
      border-radius: 8px;
      overflow: hidden;
    }}
    img {{
      max-width: 100%;
      max-height: 76vh;
      display: block;
    }}
    .meta {{
      display: grid;
      gap: 10px;
    }}
    .label-block {{
      margin-top: 16px;
    }}
    .options {{
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }}
    .option {{
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 10px 12px;
      cursor: pointer;
      background: #fff;
    }}
    .option.active {{
      border-color: #111827;
      background: #f3f4f6;
    }}
    .option-title {{
      font-weight: 700;
      margin-bottom: 4px;
    }}
    textarea {{
      width: 100%;
      min-height: 100px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 10px 12px;
      resize: vertical;
      box-sizing: border-box;
      font: inherit;
      margin-top: 8px;
    }}
    .nav {{
      display: flex;
      justify-content: space-between;
      margin-top: 16px;
      gap: 8px;
    }}
    .progress {{
      margin-top: 8px;
      font-size: 14px;
      color: #4b5563;
    }}
    .status {{
      color: #047857;
      font-weight: 600;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .image-wrap {{ min-height: 320px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div>
        <div class="title">{title}</div>
        <div class="sub" id="summary"></div>
      </div>
      <div class="actions">
        <button id="exportCsv" class="primary">Export CSV</button>
        <button id="exportJson">Export JSON</button>
        <button id="clearCurrent">Clear Current</button>
      </div>
    </div>
    <div class="layout">
      <div class="panel">
        <div class="image-wrap">
          <img id="image" alt="sample image">
        </div>
      </div>
      <div class="panel">
        <div class="meta">
          <div><strong id="indexLine"></strong></div>
          <div><strong>Question:</strong> <span id="question"></span></div>
          <div><strong>Answer:</strong> <span id="answer"></span></div>
          <div><strong>Sample ID:</strong> <span id="sampleId"></span></div>
        </div>

        <div class="label-block">
          <strong>Visual Evidence Type</strong>
          <div class="options" id="visualOptions"></div>
        </div>

        <div class="label-block">
          <strong>External Knowledge Need</strong>
          <div class="options" id="knowledgeOptions"></div>
        </div>

        <div class="label-block">
          <strong>Notes</strong>
          <textarea id="notes" placeholder="Optional notes"></textarea>
        </div>

        <div class="nav">
          <button id="prevBtn">Previous</button>
          <button id="nextBtn" class="primary">Save and Next</button>
        </div>
        <div class="progress">
          <span id="progress"></span>
          <span class="status" id="savedHint"></span>
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {data_json};
    const STORAGE_KEY = "codex_static_labels_" + DATA.dataset_name;
    const items = DATA.items;
    const visualOptions = DATA.visual_options;
    const knowledgeOptions = DATA.knowledge_options;
    let labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
    let currentIndex = 0;

    const el = {{
      image: document.getElementById("image"),
      question: document.getElementById("question"),
      answer: document.getElementById("answer"),
      sampleId: document.getElementById("sampleId"),
      indexLine: document.getElementById("indexLine"),
      summary: document.getElementById("summary"),
      progress: document.getElementById("progress"),
      savedHint: document.getElementById("savedHint"),
      notes: document.getElementById("notes"),
      visualOptions: document.getElementById("visualOptions"),
      knowledgeOptions: document.getElementById("knowledgeOptions"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      exportCsv: document.getElementById("exportCsv"),
      exportJson: document.getElementById("exportJson"),
      clearCurrent: document.getElementById("clearCurrent"),
    }};

    function getLabel(itemId) {{
      return labels[itemId] || {{
        item_id: itemId,
        visual_type_label: "",
        knowledge_level_label: "",
        label_notes: "",
        updated_at: "",
      }};
    }}

    function setLabel(itemId, patch) {{
      const next = Object.assign(getLabel(itemId), patch, {{
        updated_at: new Date().toISOString(),
      }});
      labels[itemId] = next;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
      el.savedHint.textContent = "Saved";
      setTimeout(() => {{
        if (el.savedHint.textContent === "Saved") el.savedHint.textContent = "";
      }}, 1200);
    }}

    function renderOptions(container, options, currentValue, onPick) {{
      container.innerHTML = "";
      options.forEach((opt) => {{
        const div = document.createElement("div");
        div.className = "option" + (currentValue === opt.id ? " active" : "");
        div.innerHTML = `
          <div class="option-title">${{opt.title}} (${{opt.shortcut}})</div>
          <div>${{opt.description}}</div>
        `;
        div.onclick = () => onPick(opt.id);
        container.appendChild(div);
      }});
    }}

    function refreshSummary() {{
      const done = Object.values(labels).filter(
        (row) => row.visual_type_label || row.knowledge_level_label || row.label_notes
      ).length;
      el.summary.textContent = `Total ${{items.length}} samples, labeled ${{done}}`;
    }}

    function saveCurrent() {{
      const item = items[currentIndex];
      setLabel(item.item_id, {{
        label_notes: el.notes.value.trim(),
      }});
      refreshSummary();
    }}

    function render() {{
      const item = items[currentIndex];
      const label = getLabel(item.item_id);
      el.image.src = "images/" + item.image_filename;
      el.question.textContent = item.display_question;
      el.answer.textContent = item.answer_text;
      el.sampleId.textContent = item.sample_id;
      el.indexLine.textContent = `Item ${{currentIndex + 1}} / ${{items.length}}`;
      el.progress.textContent = `Priority: ${{item.priority || "medium"}}`;
      el.notes.value = label.label_notes || "";

      renderOptions(el.visualOptions, visualOptions, label.visual_type_label, (value) => {{
        setLabel(item.item_id, {{ visual_type_label: value, label_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
      }});
      renderOptions(el.knowledgeOptions, knowledgeOptions, label.knowledge_level_label, (value) => {{
        setLabel(item.item_id, {{ knowledge_level_label: value, label_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
      }});
      refreshSummary();
    }}

    function download(filename, text, mime) {{
      const blob = new Blob([text], {{ type: mime }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    function exportCsv() {{
      const rows = [["item_id", "visual_type_label", "knowledge_level_label", "label_notes", "updated_at"]];
      items.forEach((item) => {{
        const row = getLabel(item.item_id);
        rows.push([
          row.item_id || item.item_id,
          row.visual_type_label || "",
          row.knowledge_level_label || "",
          row.label_notes || "",
          row.updated_at || "",
        ]);
      }});
      const text = rows
        .map((cols) =>
          cols
            .map((value) => {{
              const s = String(value ?? "");
              return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
            }})
            .join(",")
        )
        .join("\\n");
      download(DATA.dataset_name + "_labels.csv", text, "text/csv;charset=utf-8");
    }}

    function exportJson() {{
      const payload = items.map((item) => Object.assign({{}}, item, getLabel(item.item_id)));
      download(
        DATA.dataset_name + "_labels.json",
        JSON.stringify(payload, null, 2),
        "application/json;charset=utf-8"
      );
    }}

    el.notes.addEventListener("change", saveCurrent);
    el.prevBtn.onclick = () => {{
      saveCurrent();
      currentIndex = (currentIndex - 1 + items.length) % items.length;
      render();
    }};
    el.nextBtn.onclick = () => {{
      saveCurrent();
      currentIndex = (currentIndex + 1) % items.length;
      render();
    }};
    el.clearCurrent.onclick = () => {{
      const item = items[currentIndex];
      labels[item.item_id] = {{
        item_id: item.item_id,
        visual_type_label: "",
        knowledge_level_label: "",
        label_notes: "",
        updated_at: new Date().toISOString(),
      }};
      localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
      render();
      refreshSummary();
    }};
    el.exportCsv.onclick = exportCsv;
    el.exportJson.onclick = exportJson;

    window.addEventListener("keydown", (event) => {{
      const key = event.key.toLowerCase();
      const item = items[currentIndex];

      const visual = visualOptions.find((row) => row.shortcut.toLowerCase() === key);
      if (visual) {{
        setLabel(item.item_id, {{ visual_type_label: visual.id, label_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
        return;
      }}

      const knowledge = knowledgeOptions.find((row) => row.shortcut.toLowerCase() === key);
      if (knowledge) {{
        setLabel(item.item_id, {{ knowledge_level_label: knowledge.id, label_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
        return;
      }}

      if (key === "arrowright") el.nextBtn.click();
      if (key === "arrowleft") el.prevBtn.click();
    }});

    render();
  </script>
</body>
</html>
"""


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source_dir.parent))


def main() -> None:
    manifest_rows = read_csv(SOURCE_DIR / "manifest.csv")
    splits = split_round_robin(manifest_rows, N_SPLITS)

    if sum(len(split) for split in splits) != len(manifest_rows):
        raise RuntimeError("Split size mismatch.")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str]] = []

    for idx, split_rows in enumerate(splits, start=1):
        split_name = f"okvqa_type_label_round3_320_part{idx}"
        split_dir = OUTPUT_ROOT / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        local_rows: list[dict[str, str]] = []
        for rank, row in enumerate(split_rows, start=1):
            local_row = dict(row)
            local_row["rank"] = str(rank)
            local_row["local_image_path"] = str((images_dir / row["image_filename"]).resolve())
            local_rows.append(local_row)
            shutil.copy2(SOURCE_DIR / "images" / row["image_filename"], images_dir / row["image_filename"])

        if not local_rows:
            raise RuntimeError(f"Split {split_name} is empty.")

        write_csv(split_dir / "manifest.csv", local_rows, list(local_rows[0].keys()))
        write_csv(split_dir / "labels.csv", [], LABEL_FIELDS)
        (split_dir / "index.html").write_text(make_html(split_name, local_rows), encoding="utf-8")
        (split_dir / "README.txt").write_text(README_TEXT, encoding="utf-8")

        zip_path = OUTPUT_ROOT / f"{split_name}.zip"
        zip_dir(split_dir, zip_path)

        counts = {"high": 0, "medium": 0, "low": 0}
        for row in local_rows:
            priority = row.get("priority", "low")
            counts[priority] = counts.get(priority, 0) + 1

        summary_rows.append(
            {
                "split_name": split_name,
                "n_items": str(len(local_rows)),
                "n_high": str(counts.get("high", 0)),
                "n_medium": str(counts.get("medium", 0)),
                "n_low": str(counts.get("low", 0)),
                "dir": str(split_dir),
                "zip": str(zip_path),
            }
        )

    write_csv(
        OUTPUT_ROOT / "split_summary.csv",
        summary_rows,
        ["split_name", "n_items", "n_high", "n_medium", "n_low", "dir", "zip"],
    )

    print(f"[done] output_root={OUTPUT_ROOT}")
    for row in summary_rows:
        print(
            "[done] "
            f"{row['split_name']} items={row['n_items']} "
            f"high={row['n_high']} medium={row['n_medium']} low={row['n_low']}"
        )


if __name__ == "__main__":
    main()
