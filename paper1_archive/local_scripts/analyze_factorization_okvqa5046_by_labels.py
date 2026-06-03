from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


LABEL_FIELDS = [
    "question_type",
    "visual_structure",
    "image_dependence",
    "reasoning_operation",
    "ambiguity_flag",
    "extra_evidence",
]

PROMPT_FILES = [
    ("B_direct", "B_direct_eval.csv"),
    ("C_step_only", "C_step_only_eval.csv"),
    ("D_visual_only", "D_visual_only_eval.csv"),
    ("A_step_visual", "A_step_visual_eval.csv"),
    ("E_reply_step_only", "E_reply_step_only_eval.csv"),
    ("F_reply_step_visual", "F_reply_step_visual_eval.csv"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_base_id(sample_id: str) -> str:
    s = (sample_id or "").strip()
    if s.endswith("_A") or s.endswith("_B"):
        s = s[:-2]
    if s.startswith("okvqa_val_"):
        tail = s[len("okvqa_val_") :]
        if tail.isdigit():
            return f"okvqa_val_{int(tail)}"
    return s


QUESTION_SUFFIXES = [
    " Think step by step from visual evidence internally, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    " Think step by step internally, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    " Use visual evidence, then reply with only one short sentence in exactly this format: The answer is <short answer>.",
    " Reply step by step from visual evidence.",
    " Reply step by step.",
    " Reply step by step from visual evidence, then give the final answer in exactly this format: The answer is <short answer>.",
    " Reply step by step, then give the final answer in exactly this format: The answer is <short answer>.",
    " Reply with only one short sentence in exactly this format: The answer is <short answer>.",
]


def normalize_question(text: str) -> str:
    s = (text or "").strip()
    for suffix in QUESTION_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    s = re.sub(r"\s+", " ", s)
    return s


def to_bool01(value: str) -> int:
    return 1 if str(value).strip() == "1" else 0


def to_float(value: str) -> float:
    try:
        return float((value or "").strip() or "0")
    except ValueError:
        return 0.0


def load_eval_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        image_name = Path((row.get("image_path") or "").strip()).name
        question = normalize_question(row.get("question", ""))
        key = (image_name, question)
        if image_name and question:
            out[key] = row
    return out


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_signed_pct_points(x: float) -> str:
    return f"{x * 100:+.2f} pts"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze 6-prompt OK-VQA factorization results on a label-annotated subset."
    )
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv).expanduser().resolve()
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    label_rows = read_csv(labels_csv)
    eval_maps = {
        prompt: load_eval_map(eval_dir / fname)
        for prompt, fname in PROMPT_FILES
    }

    merged_rows: list[dict[str, str]] = []
    missing_by_prompt = Counter()
    for row in label_rows:
        image_name = (row.get("image_filename") or "").strip()
        question = normalize_question(row.get("question_text", ""))
        key = (image_name, question)
        if not image_name or not question:
            continue
        merged = dict(row)
        for prompt, _ in PROMPT_FILES:
            erow = eval_maps[prompt].get(key)
            if not erow:
                missing_by_prompt[prompt] += 1
                merged[f"{prompt}__correct"] = ""
                merged[f"{prompt}__vqa_score"] = ""
                merged[f"{prompt}__predicted_answer"] = ""
                merged[f"{prompt}__generated_text"] = ""
                continue
            merged[f"{prompt}__correct"] = erow.get("correct", "")
            merged[f"{prompt}__vqa_score"] = erow.get("vqa_score", "")
            merged[f"{prompt}__predicted_answer"] = erow.get("predicted_answer", "")
            merged[f"{prompt}__generated_text"] = erow.get("generated_text", "")
        merged_rows.append(merged)

    merged_fieldnames = list(merged_rows[0].keys()) if merged_rows else []
    write_csv(out_dir / "labels_factorization_merged.csv", merged_rows, merged_fieldnames)

    def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
        n = len(rows)
        prompt_stats = {}
        for prompt, _ in PROMPT_FILES:
            correct_n = sum(to_bool01(r.get(f"{prompt}__correct", "")) for r in rows)
            vqa_mean = sum(to_float(r.get(f"{prompt}__vqa_score", "")) for r in rows) / n if n else 0.0
            prompt_stats[prompt] = {
                "correct_n": correct_n,
                "strict_acc": correct_n / n if n else 0.0,
                "mean_vqa": vqa_mean,
            }
        best_strict_prompt = max(PROMPT_FILES, key=lambda x: prompt_stats[x[0]]["strict_acc"])[0] if n else ""
        best_vqa_prompt = max(PROMPT_FILES, key=lambda x: prompt_stats[x[0]]["mean_vqa"])[0] if n else ""
        return {
            "n": n,
            "prompt_stats": prompt_stats,
            "best_strict_prompt": best_strict_prompt,
            "best_vqa_prompt": best_vqa_prompt,
            "delta_A_minus_B": prompt_stats["A_step_visual"]["strict_acc"] - prompt_stats["B_direct"]["strict_acc"],
            "delta_C_minus_B": prompt_stats["C_step_only"]["strict_acc"] - prompt_stats["B_direct"]["strict_acc"],
            "delta_D_minus_B": prompt_stats["D_visual_only"]["strict_acc"] - prompt_stats["B_direct"]["strict_acc"],
            "delta_E_minus_C": prompt_stats["E_reply_step_only"]["strict_acc"] - prompt_stats["C_step_only"]["strict_acc"],
            "delta_F_minus_A": prompt_stats["F_reply_step_visual"]["strict_acc"] - prompt_stats["A_step_visual"]["strict_acc"],
            "delta_F_minus_E": prompt_stats["F_reply_step_visual"]["strict_acc"] - prompt_stats["E_reply_step_only"]["strict_acc"],
        }

    summary_rows: list[dict[str, str]] = []
    overall = summarize(merged_rows)
    row = {
        "group_family": "all",
        "group_value": "__all__",
        "n": str(overall["n"]),
        "best_strict_prompt": str(overall["best_strict_prompt"]),
        "best_vqa_prompt": str(overall["best_vqa_prompt"]),
        "A_minus_B": f"{overall['delta_A_minus_B']:.4f}",
        "C_minus_B": f"{overall['delta_C_minus_B']:.4f}",
        "D_minus_B": f"{overall['delta_D_minus_B']:.4f}",
        "E_minus_C": f"{overall['delta_E_minus_C']:.4f}",
        "F_minus_A": f"{overall['delta_F_minus_A']:.4f}",
        "F_minus_E": f"{overall['delta_F_minus_E']:.4f}",
    }
    for prompt, _ in PROMPT_FILES:
        stats = overall["prompt_stats"][prompt]
        row[f"{prompt}__strict_acc"] = f"{stats['strict_acc']:.4f}"
        row[f"{prompt}__mean_vqa"] = f"{stats['mean_vqa']:.4f}"
        row[f"{prompt}__correct_n"] = str(stats["correct_n"])
    summary_rows.append(row)

    for field in LABEL_FIELDS:
        grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for r in merged_rows:
            key = (r.get(field) or "").strip() or "<blank>"
            grouped[key].append(r)
        for key, rows in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            s = summarize(rows)
            row = {
                "group_family": field,
                "group_value": key,
                "n": str(s["n"]),
                "best_strict_prompt": str(s["best_strict_prompt"]),
                "best_vqa_prompt": str(s["best_vqa_prompt"]),
                "A_minus_B": f"{s['delta_A_minus_B']:.4f}",
                "C_minus_B": f"{s['delta_C_minus_B']:.4f}",
                "D_minus_B": f"{s['delta_D_minus_B']:.4f}",
                "E_minus_C": f"{s['delta_E_minus_C']:.4f}",
                "F_minus_A": f"{s['delta_F_minus_A']:.4f}",
                "F_minus_E": f"{s['delta_F_minus_E']:.4f}",
            }
            for prompt, _ in PROMPT_FILES:
                stats = s["prompt_stats"][prompt]
                row[f"{prompt}__strict_acc"] = f"{stats['strict_acc']:.4f}"
                row[f"{prompt}__mean_vqa"] = f"{stats['mean_vqa']:.4f}"
                row[f"{prompt}__correct_n"] = str(stats["correct_n"])
            summary_rows.append(row)

    summary_fields = [
        "group_family",
        "group_value",
        "n",
        "best_strict_prompt",
        "best_vqa_prompt",
        "A_minus_B",
        "C_minus_B",
        "D_minus_B",
        "E_minus_C",
        "F_minus_A",
        "F_minus_E",
    ]
    for prompt, _ in PROMPT_FILES:
        summary_fields.extend(
            [
                f"{prompt}__strict_acc",
                f"{prompt}__mean_vqa",
                f"{prompt}__correct_n",
            ]
        )
    write_csv(out_dir / "factorization_by_labels_summary.csv", summary_rows, summary_fields)

    md = []
    md.append("# Factorization By Labels (395 labeled samples)")
    md.append("")
    md.append(f"- labeled rows merged: `{overall['n']}`")
    if missing_by_prompt:
        md.append(
            "- missing eval rows by prompt: "
            + ", ".join(f"`{k}={v}`" for k, v in sorted(missing_by_prompt.items()))
        )
    md.append("")
    md.append("## Overall")
    for prompt, _ in PROMPT_FILES:
        stats = overall["prompt_stats"][prompt]
        md.append(
            f"- `{prompt}`: strict=`{fmt_pct(stats['strict_acc'])}`; mean_vqa=`{fmt_pct(stats['mean_vqa'])}`; correct_n=`{stats['correct_n']}`"
        )
    md.append(f"- best strict prompt: `{overall['best_strict_prompt']}`")
    md.append(f"- best mean-vqa prompt: `{overall['best_vqa_prompt']}`")
    md.append(
        f"- deltas: `A-B {fmt_signed_pct_points(overall['delta_A_minus_B'])}`, "
        f"`C-B {fmt_signed_pct_points(overall['delta_C_minus_B'])}`, "
        f"`D-B {fmt_signed_pct_points(overall['delta_D_minus_B'])}`, "
        f"`E-C {fmt_signed_pct_points(overall['delta_E_minus_C'])}`, "
        f"`F-A {fmt_signed_pct_points(overall['delta_F_minus_A'])}`, "
        f"`F-E {fmt_signed_pct_points(overall['delta_F_minus_E'])}`"
    )
    md.append("")

    for field in LABEL_FIELDS:
        md.append(f"## By {field}")
        field_rows = [r for r in summary_rows if r["group_family"] == field]
        for r in field_rows:
            md.append(
                f"- `{r['group_value']}` (n=`{r['n']}`): "
                f"best strict=`{r['best_strict_prompt']}`; "
                f"B=`{fmt_pct(float(r['B_direct__strict_acc']))}`; "
                f"C=`{fmt_pct(float(r['C_step_only__strict_acc']))}`; "
                f"D=`{fmt_pct(float(r['D_visual_only__strict_acc']))}`; "
                f"A=`{fmt_pct(float(r['A_step_visual__strict_acc']))}`; "
                f"E=`{fmt_pct(float(r['E_reply_step_only__strict_acc']))}`; "
                f"F=`{fmt_pct(float(r['F_reply_step_visual__strict_acc']))}`; "
                f"deltas=`A-B {fmt_signed_pct_points(float(r['A_minus_B']))}`, "
                f"`C-B {fmt_signed_pct_points(float(r['C_minus_B']))}`, "
                f"`D-B {fmt_signed_pct_points(float(r['D_minus_B']))}`, "
                f"`E-C {fmt_signed_pct_points(float(r['E_minus_C']))}`, "
                f"`F-A {fmt_signed_pct_points(float(r['F_minus_A']))}`, "
                f"`F-E {fmt_signed_pct_points(float(r['F_minus_E']))}`"
            )
        md.append("")

    write_text(out_dir / "factorization_by_labels_summary.md", "\n".join(md) + "\n")
    print(f"[done] merged={overall['n']}")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
