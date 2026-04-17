#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def _norm(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_prompt_suffix(question: str) -> str:
    q = (question or "").strip()
    suffixes = [
        " Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>.",
        " Reply exactly in the format: The answer is <short answer>.",
    ]
    for suf in suffixes:
        if q.endswith(suf):
            return q[: -len(suf)].strip()
    return q


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_ann_map(path: Path) -> dict[str, list[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for item in obj.get("annotations", []):
        qid = str(item.get("question_id", "")).strip()
        if not qid:
            continue
        ans = []
        for a in item.get("answers", []) or []:
            x = (a.get("answer", "") if isinstance(a, dict) else "").strip()
            if x:
                ans.append(x)
        if ans:
            out[qid] = ans
    return out


def _hit(pred: str, gold_list: list[str]) -> bool:
    p = _norm(pred)
    if not p:
        return False
    for g in gold_list:
        gn = _norm(g)
        if gn and gn in p:
            return True
    return False


def _bucket(a_hit: bool, b_hit: bool) -> str:
    return f"A{int(a_hit)}_B{int(b_hit)}"


def _sid_key(sid: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", sid or "")
    if m:
        return (int(m.group(1)), sid or "")
    return (10**12, sid or "")


def _map_image_path(path: str, src_prefix: str, dst_prefix: str) -> str:
    p = (path or "").strip()
    if not p:
        return p
    if src_prefix and dst_prefix and p.startswith(src_prefix):
        return dst_prefix + p[len(src_prefix) :]
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full A/B hit-based comparison CSV + markdown report.")
    parser.add_argument("--run-a-csv", required=True)
    parser.add_argument("--run-b-csv", required=True)
    parser.add_argument("--annotations-json", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-per-bucket", type=int, default=60)
    parser.add_argument("--title", default="A/B Buckets (Hit Only)")
    parser.add_argument(
        "--image-prefix-from",
        default="",
        help="Optional source path prefix to replace in image_path (e.g. /home/xtyu/tca-reasoning/data/okvqa/images/val2014/)",
    )
    parser.add_argument(
        "--image-prefix-to",
        default="",
        help="Optional destination path prefix for local rendering (e.g. D:/code/Bridging/data/okvqa/images/val2014/)",
    )
    parser.add_argument(
        "--write-combined-md",
        action="store_true",
        help="Also write a single combined markdown file (can be very large).",
    )
    args = parser.parse_args()

    run_a_csv = Path(args.run_a_csv).expanduser().resolve()
    run_b_csv = Path(args.run_b_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_a = _read_csv(run_a_csv)
    rows_b = _read_csv(run_b_csv)
    if not rows_a or not rows_b:
        raise ValueError("empty A/B csv")

    a_map = {(r.get("sample_id") or "").strip(): r for r in rows_a if (r.get("sample_id") or "").strip()}
    b_map = {(r.get("sample_id") or "").strip(): r for r in rows_b if (r.get("sample_id") or "").strip()}
    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()), key=_sid_key)
    if not common_ids:
        raise ValueError("no overlapping sample_id")

    ann_map: dict[str, list[str]] = {}
    if args.annotations_json:
        ann_map = _load_ann_map(Path(args.annotations_json).expanduser().resolve())

    all_rows: list[dict] = []
    counts = Counter()
    for sid in common_ids:
        a = a_map[sid]
        b = b_map[sid]
        qid = (a.get("question_id") or b.get("question_id") or "").strip()
        iid = (a.get("image_id") or b.get("image_id") or "").strip()
        gold_fallback = (a.get("gold_answer") or b.get("gold_answer") or "").strip()
        gold_list = ann_map.get(qid, [])
        if not gold_list and gold_fallback:
            gold_list = [gold_fallback]

        a_raw = (a.get("generated_text") or "").strip()
        b_raw = (b.get("generated_text") or "").strip()
        a_pred = (a.get("predicted_answer") or "").strip()
        b_pred = (b.get("predicted_answer") or "").strip()

        ah = _hit(a_pred if a_pred else a_raw, gold_list)
        bh = _hit(b_pred if b_pred else b_raw, gold_list)
        bkt = _bucket(ah, bh)
        counts[bkt] += 1

        a_input = (a.get("question") or "").strip()
        b_input = (b.get("question") or "").strip()
        orig = _strip_prompt_suffix(a_input if a_input else b_input)

        img_path = _map_image_path(
            (a.get("image_path") or b.get("image_path") or "").strip(),
            args.image_prefix_from,
            args.image_prefix_to,
        )

        all_rows.append(
            {
                "sample_id": sid,
                "bucket": bkt,
                "a_hit": "1" if ah else "0",
                "b_hit": "1" if bh else "0",
                "question_id": qid,
                "image_id": iid,
                "image_path": img_path,
                "original_question": orig,
                "a_input": a_input,
                "b_input": b_input,
                "gold_answers": " | ".join(gold_list),
                "a_raw": a_raw,
                "b_raw": b_raw,
                "a_pred": a_pred,
                "b_pred": b_pred,
            }
        )

    full_csv = out_dir / "ab_buckets_by_hit_from_latest.csv"
    with full_csv.open("w", encoding="utf-8", newline="") as f:
        fields = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    for bkt in ["A1_B0", "A0_B1", "A1_B1", "A0_B0"]:
        part = [r for r in all_rows if r["bucket"] == bkt]
        p = out_dir / f"ab_candidates_{bkt}.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            fields = list(all_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(part)

    def _render_bucket_section(title: str, bkt: str) -> list[str]:
        out: list[str] = []
        out.append(f"## {title}")
        out.append("")
        rows = [r for r in all_rows if r["bucket"] == bkt]
        rows = rows[: max(1, args.top_per_bucket)]
        for i, r in enumerate(rows, 1):
            sid = r["sample_id"]
            img = r["image_path"]
            out.append(f"### {i}. `{sid}`")
            if img:
                out.append(f"![{sid}]({img})")
            out.append(f"- Original Question: {r['original_question']}")
            out.append(f"- A Input: {r['a_input']}")
            out.append(f"- B Input: {r['b_input']}")
            out.append(f"- Gold (all): {r['gold_answers']}")
            out.append(f"- A raw: {r['a_raw']}")
            out.append(f"- A hit: {r['a_hit']}")
            out.append(f"- B raw: {r['b_raw']}")
            out.append(f"- B hit: {r['b_hit']}")
            out.append("")
        return out

    sections = [
        ("Priority: A0_B1 (B hit, A miss)", "A0_B1"),
        ("Priority: A1_B0 (A hit, B miss)", "A1_B0"),
        ("Reference: A1_B1 (both hit)", "A1_B1"),
        ("Reference: A0_B0 (both miss)", "A0_B0"),
    ]

    # Write per-bucket markdown files (default path for easier rendering).
    split_md_paths = []
    for title, bkt in sections:
        md_lines: list[str] = []
        md_lines.append(f"# {args.title} - {bkt}")
        md_lines.append("")
        md_lines.append("- Rule: `hit = any(normalized_gt in normalized_prediction_or_raw_output)`")
        md_lines.append("")
        md_lines.append("## Bucket Counts (by hit)")
        for bb in ["A1_B1", "A1_B0", "A0_B1", "A0_B0"]:
            md_lines.append(f"- {bb}: {counts[bb]}")
        md_lines.append("")
        md_lines.extend(_render_bucket_section(title, bkt))
        p = out_dir / f"ab_bucket_{bkt}_hit_only_latest.md"
        p.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
        split_md_paths.append(p)

    # Optional combined markdown.
    md_path = out_dir / "ab_bucket_examples_hit_only_latest.md"
    if args.write_combined_md:
        lines: list[str] = []
        lines.append(f"# {args.title}")
        lines.append("")
        lines.append("- Rule: `hit = any(normalized_gt in normalized_prediction_or_raw_output)`")
        lines.append("")
        lines.append("## Bucket Counts (by hit)")
        for b in ["A1_B1", "A1_B0", "A0_B1", "A0_B0"]:
            lines.append(f"- {b}: {counts[b]}")
        lines.append("")
        for title, bkt in sections:
            lines.extend(_render_bucket_section(title, bkt))
        md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"[done] common_ids={len(common_ids)}")
    print(f"[done] full_csv={full_csv}")
    for p in split_md_paths:
        print(f"[done] md_split={p}")
    if args.write_combined_md:
        print(f"[done] md={md_path}")
    for b in ["A1_B1", "A1_B0", "A0_B1", "A0_B0"]:
        print(f"[count] {b}={counts[b]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
