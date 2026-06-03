#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(r"E:\Bridging")
DATASET_DIR = ROOT / "annotation" / "okvqa_type_label_round3_320"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
HTML_OUT = DATASET_DIR / "five_axis_annotation_ui.html"
CSV_TEMPLATE_OUT = DATASET_DIR / "five_axis_annotation_blank.csv"
README_OUT = DATASET_DIR / "five_axis_annotation_ui_README.txt"


QUESTION_TYPE_OPTIONS = [
    {
        "id": "direct_visual",
        "title": "Direct Visual",
        "shortcut": "1",
        "description": "Mainly asks for directly visible content in the image.",
    },
    {
        "id": "identity_or_subtype_recognition",
        "title": "Identity / Subtype",
        "shortcut": "2",
        "description": "Asks for a more specific identity, subtype, brand, model, or place.",
    },
    {
        "id": "entity_fact_lookup",
        "title": "Entity Fact",
        "shortcut": "3",
        "description": "Recognize the entity, then answer with an external fact about it.",
    },
    {
        "id": "situation_inference",
        "title": "Situation Inference",
        "shortcut": "4",
        "description": "Requires inferring context, event, behavior, or likely state.",
    },
    {
        "id": "relation_reasoning",
        "title": "Relation Reasoning",
        "shortcut": "5",
        "description": "Answer depends on relations among multiple entities or parts.",
    },
]

VISUAL_STRUCTURE_OPTIONS = [
    {
        "id": "single_core",
        "title": "Single Core",
        "shortcut": "Q",
        "description": "One concentrated core object or region is enough.",
    },
    {
        "id": "split_cores",
        "title": "Split Cores",
        "shortcut": "W",
        "description": "Need two or more separated core regions.",
    },
    {
        "id": "core_plus_context",
        "title": "Core + Context",
        "shortcut": "E",
        "description": "A main core cue exists, but surrounding context also matters.",
    },
    {
        "id": "diffuse_global",
        "title": "Diffuse / Global",
        "shortcut": "R",
        "description": "Evidence is spread across the broader scene without one dominant core.",
    },
]

IMAGE_DEPENDENCE_OPTIONS = [
    {
        "id": "strong",
        "title": "Strong",
        "shortcut": "A",
        "description": "Without the image, the answer is hard or impossible to recover.",
    },
    {
        "id": "mixed",
        "title": "Mixed",
        "shortcut": "S",
        "description": "Image gives the main clue, but commonsense or background knowledge completes the answer.",
    },
    {
        "id": "weak",
        "title": "Weak",
        "shortcut": "D",
        "description": "Image mostly anchors the topic; answer relies weakly on image details.",
    },
]

REASONING_OPERATION_OPTIONS = [
    {
        "id": "visual_readout",
        "title": "Visual Readout",
        "shortcut": "Z",
        "description": "Directly read visible properties such as color, count, pose, or position.",
    },
    {
        "id": "entity_linking",
        "title": "Entity Linking",
        "shortcut": "X",
        "description": "Identify an entity and map it to a known category, brand, type, or place.",
    },
    {
        "id": "world_fact_retrieval",
        "title": "World Fact Retrieval",
        "shortcut": "C",
        "description": "Recognize the entity and retrieve a fact about it.",
    },
    {
        "id": "commonsense_affordance",
        "title": "Commonsense / Affordance",
        "shortcut": "V",
        "description": "Use object function, behavior, or practical commonsense.",
    },
    {
        "id": "scene_inference",
        "title": "Scene Inference",
        "shortcut": "B",
        "description": "Infer event, intent, time, state, or broader scene meaning.",
    },
    {
        "id": "relation_reasoning",
        "title": "Relation Reasoning",
        "shortcut": "N",
        "description": "Reason over relations among multiple objects, people, or regions.",
    },
    {
        "id": "symbol_text_reading",
        "title": "Symbol / Text Reading",
        "shortcut": "M",
        "description": "Use OCR, logo, sign, flag, or symbol recognition.",
    },
]

AMBIGUITY_OPTIONS = [
    {
        "id": "low",
        "title": "Low",
        "shortcut": "7",
        "description": "The labels feel fairly clear and stable.",
    },
    {
        "id": "medium",
        "title": "Medium",
        "shortcut": "8",
        "description": "There is some uncertainty, but one label still feels more plausible.",
    },
    {
        "id": "high",
        "title": "High",
        "shortcut": "9",
        "description": "Hard to classify cleanly; this is a genuinely ambiguous case.",
    },
]

LABEL_FIELDS = [
    "item_id",
    "sample_id",
    "priority",
    "image_filename",
    "question_text",
    "answer_text",
    "question_type",
    "visual_structure",
    "image_dependence",
    "reasoning_operation",
    "ambiguity_flag",
    "annotator_notes",
    "updated_at",
]

README_TEXT = """Five-Axis Annotation UI
======================

This UI is fully offline. No Python or local server is required after this file has been created.

How to use
----------
1. Open `five_axis_annotation_ui.html` in a browser.
2. For each sample, annotate:
   - Question Type
   - Visual Structure
   - Image Dependence
   - Reasoning Operation
   - Ambiguity Flag
3. Optional: add notes.
4. Click `Export CSV` when finished.

Keyboard shortcuts
------------------
- Question Type: 1 / 2 / 3 / 4 / 5
- Visual Structure: Q / W / E / R
- Image Dependence: A / S / D
- Reasoning Operation: Z / X / C / V / B / N / M
- Ambiguity: 7 / 8 / 9
- Navigation: Left / Right arrow
"""


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_payload(rows: list[dict[str, str]]) -> dict:
    items = []
    for row in rows:
        items.append(
            {
                "item_id": row["item_id"],
                "sample_id": row["sample_id"],
                "priority": row["priority"],
                "question_text": row["display_question"].strip(),
                "answer_text": row["answer_text"].strip(),
                "image_filename": row["image_filename"],
            }
        )
    return {
        "dataset_name": "okvqa_type_label_round3_320_five_axis",
        "items": items,
        "question_type_options": QUESTION_TYPE_OPTIONS,
        "visual_structure_options": VISUAL_STRUCTURE_OPTIONS,
        "image_dependence_options": IMAGE_DEPENDENCE_OPTIONS,
        "reasoning_operation_options": REASONING_OPERATION_OPTIONS,
        "ambiguity_options": AMBIGUITY_OPTIONS,
    }


def make_html(payload: dict) -> str:
    title = html.escape(payload["dataset_name"])
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
      max-width: 1500px;
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
      line-height: 1.4;
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
      grid-template-columns: 1fr 1.05fr;
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
    .section {{
      margin-top: 16px;
    }}
    .section-title {{
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .options {{
      display: grid;
      gap: 8px;
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
      min-height: 110px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 10px 12px;
      resize: vertical;
      box-sizing: border-box;
      font: inherit;
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
      margin-left: 12px;
    }}
    @media (max-width: 1100px) {{
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
        <div class="sub">Shortcuts: question type 1-5, visual structure Q/W/E/R, image dependence A/S/D, reasoning op Z/X/C/V/B/N/M, ambiguity 7/8/9, arrows for navigation.</div>
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

        <div class="section">
          <div class="section-title">Question Type</div>
          <div class="options" id="questionTypeOptions"></div>
        </div>

        <div class="section">
          <div class="section-title">Visual Structure</div>
          <div class="options" id="visualStructureOptions"></div>
        </div>

        <div class="section">
          <div class="section-title">Image Dependence</div>
          <div class="options" id="imageDependenceOptions"></div>
        </div>

        <div class="section">
          <div class="section-title">Reasoning Operation</div>
          <div class="options" id="reasoningOperationOptions"></div>
        </div>

        <div class="section">
          <div class="section-title">Ambiguity Flag</div>
          <div class="options" id="ambiguityOptions"></div>
        </div>

        <div class="section">
          <div class="section-title">Notes</div>
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
    const STORAGE_KEY = "codex_five_axis_" + DATA.dataset_name;
    const items = DATA.items;
    const questionTypeOptions = DATA.question_type_options;
    const visualStructureOptions = DATA.visual_structure_options;
    const imageDependenceOptions = DATA.image_dependence_options;
    const reasoningOperationOptions = DATA.reasoning_operation_options;
    const ambiguityOptions = DATA.ambiguity_options;
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
      questionTypeOptions: document.getElementById("questionTypeOptions"),
      visualStructureOptions: document.getElementById("visualStructureOptions"),
      imageDependenceOptions: document.getElementById("imageDependenceOptions"),
      reasoningOperationOptions: document.getElementById("reasoningOperationOptions"),
      ambiguityOptions: document.getElementById("ambiguityOptions"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      exportCsv: document.getElementById("exportCsv"),
      exportJson: document.getElementById("exportJson"),
      clearCurrent: document.getElementById("clearCurrent"),
    }};

    function getLabel(itemId) {{
      return labels[itemId] || {{
        item_id: itemId,
        question_type: "",
        visual_structure: "",
        image_dependence: "",
        reasoning_operation: "",
        ambiguity_flag: "",
        annotator_notes: "",
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
        (row) =>
          row.question_type ||
          row.visual_structure ||
          row.image_dependence ||
          row.reasoning_operation ||
          row.ambiguity_flag ||
          row.annotator_notes
      ).length;
      el.summary.textContent = `Total ${{items.length}} samples, labeled ${{done}}`;
    }}

    function saveCurrent() {{
      const item = items[currentIndex];
      setLabel(item.item_id, {{
        annotator_notes: el.notes.value.trim(),
      }});
      refreshSummary();
    }}

    function render() {{
      const item = items[currentIndex];
      const label = getLabel(item.item_id);
      el.image.src = "images/" + item.image_filename;
      el.question.textContent = item.question_text;
      el.answer.textContent = item.answer_text;
      el.sampleId.textContent = item.sample_id;
      el.indexLine.textContent = `Item ${{currentIndex + 1}} / ${{items.length}}`;
      el.progress.textContent = `Priority: ${{item.priority || "medium"}}`;
      el.notes.value = label.annotator_notes || "";

      renderOptions(el.questionTypeOptions, questionTypeOptions, label.question_type, (value) => {{
        setLabel(item.item_id, {{ question_type: value, annotator_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
      }});
      renderOptions(el.visualStructureOptions, visualStructureOptions, label.visual_structure, (value) => {{
        setLabel(item.item_id, {{ visual_structure: value, annotator_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
      }});
      renderOptions(el.imageDependenceOptions, imageDependenceOptions, label.image_dependence, (value) => {{
        setLabel(item.item_id, {{ image_dependence: value, annotator_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
      }});
      renderOptions(el.reasoningOperationOptions, reasoningOperationOptions, label.reasoning_operation, (value) => {{
        setLabel(item.item_id, {{ reasoning_operation: value, annotator_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
      }});
      renderOptions(el.ambiguityOptions, ambiguityOptions, label.ambiguity_flag, (value) => {{
        setLabel(item.item_id, {{ ambiguity_flag: value, annotator_notes: el.notes.value.trim() }});
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
      const rows = [[
        "item_id",
        "sample_id",
        "priority",
        "image_filename",
        "question_text",
        "answer_text",
        "question_type",
        "visual_structure",
        "image_dependence",
        "reasoning_operation",
        "ambiguity_flag",
        "annotator_notes",
        "updated_at"
      ]];
      items.forEach((item) => {{
        const row = getLabel(item.item_id);
        rows.push([
          item.item_id,
          item.sample_id,
          item.priority,
          item.image_filename,
          item.question_text,
          item.answer_text,
          row.question_type || "",
          row.visual_structure || "",
          row.image_dependence || "",
          row.reasoning_operation || "",
          row.ambiguity_flag || "",
          row.annotator_notes || "",
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
        question_type: "",
        visual_structure: "",
        image_dependence: "",
        reasoning_operation: "",
        ambiguity_flag: "",
        annotator_notes: "",
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

      const questionType = questionTypeOptions.find((row) => row.shortcut.toLowerCase() === key);
      if (questionType) {{
        setLabel(item.item_id, {{ question_type: questionType.id, annotator_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
        return;
      }}

      const visualStructure = visualStructureOptions.find((row) => row.shortcut.toLowerCase() === key);
      if (visualStructure) {{
        setLabel(item.item_id, {{ visual_structure: visualStructure.id, annotator_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
        return;
      }}

      const imageDependence = imageDependenceOptions.find((row) => row.shortcut.toLowerCase() === key);
      if (imageDependence) {{
        setLabel(item.item_id, {{ image_dependence: imageDependence.id, annotator_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
        return;
      }}

      const reasoningOperation = reasoningOperationOptions.find((row) => row.shortcut.toLowerCase() === key);
      if (reasoningOperation) {{
        setLabel(item.item_id, {{ reasoning_operation: reasoningOperation.id, annotator_notes: el.notes.value.trim() }});
        render();
        refreshSummary();
        return;
      }}

      const ambiguity = ambiguityOptions.find((row) => row.shortcut.toLowerCase() === key);
      if (ambiguity) {{
        setLabel(item.item_id, {{ ambiguity_flag: ambiguity.id, annotator_notes: el.notes.value.trim() }});
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


def main() -> None:
    rows = read_manifest(MANIFEST_PATH)
    payload = build_payload(rows)
    HTML_OUT.write_text(make_html(payload), encoding="utf-8")

    blank_rows = []
    for item in payload["items"]:
        blank_rows.append(
            {
                "item_id": item["item_id"],
                "sample_id": item["sample_id"],
                "priority": item["priority"],
                "image_filename": item["image_filename"],
                "question_text": item["question_text"],
                "answer_text": item["answer_text"],
                "question_type": "",
                "visual_structure": "",
                "image_dependence": "",
                "reasoning_operation": "",
                "ambiguity_flag": "",
                "annotator_notes": "",
                "updated_at": "",
            }
        )

    write_csv(CSV_TEMPLATE_OUT, blank_rows, LABEL_FIELDS)
    README_OUT.write_text(README_TEXT, encoding="utf-8")

    print(f"[done] html={HTML_OUT}")
    print(f"[done] csv_template={CSV_TEMPLATE_OUT}")
    print(f"[done] readme={README_OUT}")


if __name__ == "__main__":
    main()
