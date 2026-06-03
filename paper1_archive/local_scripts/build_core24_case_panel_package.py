#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CASE_ORDER = [
    ("okvqa_val_3658865", "support_oriented"),
    ("okvqa_val_2847255", "suppressor_route"),
    ("okvqa_val_340155", "mixed_suppressor_behavior"),
    ("okvqa_val_02444", "mixed_visual_readout"),
]


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser().resolve())


def _fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _run_name(prompt: str) -> str:
    if prompt == "B_direct":
        return "B_direct"
    if prompt == "D_visual_only":
        return "D_visual_only"
    return str(prompt)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure-ready case panel package for core24 region evidence cases.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--wrong-image-detail-csv", required=True)
    parser.add_argument("--region-case-csv", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest = _read(args.manifest_csv)
    wrong = _read(args.wrong_image_detail_csv)
    region = _read(args.region_case_csv)
    region["run_name"] = region["prompt_name"].map(_run_name)

    rows = []
    for rank, (sample_id, case_role) in enumerate(CASE_ORDER, start=1):
        meta = manifest[manifest["sample_id"] == sample_id].head(1)
        if meta.empty:
            continue
        meta_row = meta.iloc[0].to_dict()
        stem = Path(str(meta_row["image_filename"])).stem
        image_path = str(Path(str(meta_row["local_image_path"])).resolve())
        answer_mask_path = str((Path(args.mask_root).resolve() / stem / "answer.png"))
        relate_mask_path = str((Path(args.mask_root).resolve() / stem / "relate.png"))

        prompts = sorted(set(region.loc[region["sample_id"] == sample_id, "run_name"]).union(set(wrong.loc[wrong["sample_id"] == sample_id, "run_name"])))
        for prompt in prompts:
            r = region[(region["sample_id"] == sample_id) & (region["run_name"] == prompt)].head(1)
            w = wrong[(wrong["sample_id"] == sample_id) & (wrong["run_name"] == prompt)].head(1)
            rr = r.iloc[0].to_dict() if not r.empty else {}
            ww = w.iloc[0].to_dict() if not w.empty else {}
            rows.append(
                {
                    "case_rank": rank,
                    "case_role": case_role,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "question": meta_row.get("question_text", ""),
                    "gold_answer": meta_row.get("answer_text", ""),
                    "reasoning_operation": meta_row.get("reasoning_operation", ""),
                    "visual_structure": meta_row.get("visual_structure", ""),
                    "image_path": image_path,
                    "answer_mask_path": answer_mask_path,
                    "relate_mask_path": relate_mask_path,
                    "clean_answer": ww.get("clean_predicted_answer", ""),
                    "wrong_image_answer": ww.get("wrong_image_predicted_answer", ""),
                    "wrong_image_target_rank": ww.get("wrong_image_target_rank", ""),
                    "wrong_image_margin_drop": ww.get("margin_drop_vs_clean", ""),
                    "wrong_support_weakening": ww.get("support_route_weakening", ""),
                    "wrong_suppressor_weakening": ww.get("suppressor_route_weakening", ""),
                    "answer_rank_damage_over_random4": rr.get("answer_rank_damage_over_random4", ""),
                    "union_rank_damage_over_random4": rr.get("union_rank_damage_over_random4", ""),
                    "answer_margin_drop_over_random4": rr.get("answer_margin_drop_over_random4", ""),
                    "region_support_answer_weakening": rr.get("weakening_support_source_answer_mask", ""),
                    "region_support_union_weakening": rr.get("weakening_support_source_union_mask", ""),
                    "region_suppressor_answer_weakening": rr.get("weakening_suppressor_source_answer_mask", ""),
                    "region_suppressor_union_weakening": rr.get("weakening_suppressor_source_union_mask", ""),
                    "case_profile": rr.get("case_profile", ""),
                }
            )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "core24_case_panel_data.csv", index=False)

    lines = [
        "# Core24 Case Panel Package",
        "",
        "This package fixes the data fields for the current report-ready case panels.",
        "",
    ]
    for rank, (sample_id, case_role) in enumerate(CASE_ORDER, start=1):
        sub = out[out["sample_id"] == sample_id]
        if sub.empty:
            continue
        first = sub.iloc[0]
        lines.extend(
            [
                f"## {rank}. `{sample_id}`",
                "",
                f"- role: `{case_role}`",
                f"- question: {first['question']}",
                f"- gold answer: `{first['gold_answer']}`",
                f"- type: `{first['reasoning_operation']}` / `{first['visual_structure']}`",
                f"- image: `{first['image_path']}`",
                f"- answer mask: `{first['answer_mask_path']}`",
                f"- relate mask: `{first['relate_mask_path']}`",
                "",
                "| prompt | clean answer | wrong-image answer | wrong rank | wrong margin drop | wrong support weak | wrong suppressor weak | answer rank over random4 | union rank over random4 | region support answer | region suppressor answer |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in sub.iterrows():
            lines.append(
                "| {prompt} | {clean} | {wrong} | {wrong_rank} | {wrong_margin} | {wrong_support} | {wrong_suppressor} | {answer_rank} | {union_rank} | {region_support} | {region_suppressor} |".format(
                    prompt=row["prompt"],
                    clean=_fmt(row["clean_answer"]),
                    wrong=_fmt(row["wrong_image_answer"]),
                    wrong_rank=_fmt(row["wrong_image_target_rank"]),
                    wrong_margin=_fmt(row["wrong_image_margin_drop"]),
                    wrong_support=_fmt(row["wrong_support_weakening"]),
                    wrong_suppressor=_fmt(row["wrong_suppressor_weakening"]),
                    answer_rank=_fmt(row["answer_rank_damage_over_random4"]),
                    union_rank=_fmt(row["union_rank_damage_over_random4"]),
                    region_support=_fmt(row["region_support_answer_weakening"]),
                    region_suppressor=_fmt(row["region_suppressor_answer_weakening"]),
                )
            )
        lines.append("")
    (out_dir / "CORE24_CASE_PANEL_PACKAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
