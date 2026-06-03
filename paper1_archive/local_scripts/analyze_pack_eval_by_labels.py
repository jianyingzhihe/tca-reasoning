from __future__ import annotations

import argparse
import csv
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


def to_bool01(value: str) -> int:
    return 1 if str(value).strip() == "1" else 0


def sample_base_id(sample_id: str) -> str:
    s = (sample_id or "").strip()
    if s.endswith("_A") or s.endswith("_B"):
        return s[:-2]
    return s


def load_eval_map(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        base_id = sample_base_id(row.get("sample_id", ""))
        if base_id:
            out[base_id] = row
    return out


def fmt_rate(num: int, den: int) -> str:
    if den == 0:
        return ""
    return f"{num / den:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Prompt A/B eval outputs by label fields for a stratified pack."
    )
    parser.add_argument("--pack-csv", required=True, help="Label-enriched pack csv")
    parser.add_argument("--eval-a-csv", required=True, help="Prompt A eval csv")
    parser.add_argument("--eval-b-csv", required=True, help="Prompt B eval csv")
    parser.add_argument("--bucket-csv", required=True, help="A/B bucket csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    pack_rows = read_csv(Path(args.pack_csv).expanduser().resolve())
    eval_a = load_eval_map(Path(args.eval_a_csv).expanduser().resolve())
    eval_b = load_eval_map(Path(args.eval_b_csv).expanduser().resolve())
    bucket_rows = read_csv(Path(args.bucket_csv).expanduser().resolve())
    bucket_map = {row.get("sample_id", "").strip(): row for row in bucket_rows}
    out_dir = Path(args.out_dir).expanduser().resolve()

    merged_rows: list[dict[str, str]] = []
    for row in pack_rows:
        sid = (row.get("sample_id") or "").strip()
        if not sid:
            continue
        a_row = eval_a.get(sid, {})
        b_row = eval_b.get(sid, {})
        bucket_row = bucket_map.get(sid, {})
        merged = dict(row)
        merged["a_correct"] = a_row.get("correct", "")
        merged["b_correct"] = b_row.get("correct", "")
        merged["a_vqa_score"] = a_row.get("vqa_score", "")
        merged["b_vqa_score"] = b_row.get("vqa_score", "")
        merged["a_predicted_answer"] = a_row.get("predicted_answer", "")
        merged["b_predicted_answer"] = b_row.get("predicted_answer", "")
        merged["a_generated_text"] = a_row.get("generated_text", "")
        merged["b_generated_text"] = b_row.get("generated_text", "")
        merged["bucket"] = bucket_row.get("bucket", "")
        merged_rows.append(merged)

    merged_fieldnames = list(merged_rows[0].keys()) if merged_rows else []
    write_csv(out_dir / "pack_eval_merged.csv", merged_rows, merged_fieldnames)

    summary_rows: list[dict[str, str]] = []
    bucket_counter = Counter(row.get("bucket", "") for row in merged_rows)
    overall_n = len(merged_rows)
    overall_a = sum(to_bool01(row.get("a_correct", "")) for row in merged_rows)
    overall_b = sum(to_bool01(row.get("b_correct", "")) for row in merged_rows)
    summary_rows.append(
        {
            "group_family": "all",
            "group_value": "__all__",
            "n": str(overall_n),
            "a_correct_n": str(overall_a),
            "b_correct_n": str(overall_b),
            "a_acc": fmt_rate(overall_a, overall_n),
            "b_acc": fmt_rate(overall_b, overall_n),
            "a_minus_b": f"{(overall_a / overall_n - overall_b / overall_n):.3f}" if overall_n else "",
            "A1_B1": str(bucket_counter.get("A1_B1", 0)),
            "A1_B0": str(bucket_counter.get("A1_B0", 0)),
            "A0_B1": str(bucket_counter.get("A0_B1", 0)),
            "A0_B0": str(bucket_counter.get("A0_B0", 0)),
        }
    )

    for field in LABEL_FIELDS:
        groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for row in merged_rows:
            key = (row.get(field) or "").strip() or "<blank>"
            groups[key].append(row)
        for key, rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            n = len(rows)
            a_n = sum(to_bool01(row.get("a_correct", "")) for row in rows)
            b_n = sum(to_bool01(row.get("b_correct", "")) for row in rows)
            bc = Counter(row.get("bucket", "") for row in rows)
            summary_rows.append(
                {
                    "group_family": field,
                    "group_value": key,
                    "n": str(n),
                    "a_correct_n": str(a_n),
                    "b_correct_n": str(b_n),
                    "a_acc": fmt_rate(a_n, n),
                    "b_acc": fmt_rate(b_n, n),
                    "a_minus_b": f"{(a_n / n - b_n / n):.3f}" if n else "",
                    "A1_B1": str(bc.get("A1_B1", 0)),
                    "A1_B0": str(bc.get("A1_B0", 0)),
                    "A0_B1": str(bc.get("A0_B1", 0)),
                    "A0_B0": str(bc.get("A0_B0", 0)),
                }
            )

    summary_fields = [
        "group_family",
        "group_value",
        "n",
        "a_correct_n",
        "b_correct_n",
        "a_acc",
        "b_acc",
        "a_minus_b",
        "A1_B1",
        "A1_B0",
        "A0_B1",
        "A0_B0",
    ]
    write_csv(out_dir / "pack_eval_label_summary.csv", summary_rows, summary_fields)

    md_lines = [
        "# Pack Eval By Labels",
        "",
        f"- merged samples: `{overall_n}`",
        f"- Prompt A correct: `{overall_a}` (`{fmt_rate(overall_a, overall_n)}`)",
        f"- Prompt B correct: `{overall_b}` (`{fmt_rate(overall_b, overall_n)}`)",
        f"- A minus B: `{(overall_a / overall_n - overall_b / overall_n):.3f}`" if overall_n else "- A minus B: ``",
        "",
        "## Overall bucket mix",
        f"- `A1_B1`: `{bucket_counter.get('A1_B1', 0)}`",
        f"- `A1_B0`: `{bucket_counter.get('A1_B0', 0)}`",
        f"- `A0_B1`: `{bucket_counter.get('A0_B1', 0)}`",
        f"- `A0_B0`: `{bucket_counter.get('A0_B0', 0)}`",
        "",
    ]

    for field in LABEL_FIELDS[:-1]:
        md_lines.append(f"## By {field}")
        rows = [r for r in summary_rows if r["group_family"] == field]
        for row in rows:
            md_lines.append(
                f"- `{row['group_value']}`: "
                f"n=`{row['n']}`, "
                f"A=`{row['a_acc']}`, "
                f"B=`{row['b_acc']}`, "
                f"A-B=`{row['a_minus_b']}`, "
                f"bucket=(`{row['A1_B1']}`, `{row['A1_B0']}`, `{row['A0_B1']}`, `{row['A0_B0']}`)"
            )
        md_lines.append("")

    write_text(out_dir / "pack_eval_label_summary.md", "\n".join(md_lines) + "\n")
    print(f"[done] merged={overall_n}")
    print(f"[done] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
