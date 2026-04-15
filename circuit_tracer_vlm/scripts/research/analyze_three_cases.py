#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from circuit_tracer.graph import Graph


@dataclass
class CaseRow:
    sample_id: str
    status: str
    replacement_score: float
    completeness_score: float
    n_nodes_total: int
    n_nonzero_edges: int


def _read_metrics(path: Path) -> list[CaseRow]:
    out: list[CaseRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            status = (r.get("status") or "").strip()
            if status != "ok":
                continue
            out.append(
                CaseRow(
                    sample_id=(r.get("sample_id") or "").strip(),
                    status=status,
                    replacement_score=float(r["replacement_score"]),
                    completeness_score=float(r["completeness_score"]),
                    n_nodes_total=int(float(r.get("n_nodes_total", 0) or 0)),
                    n_nonzero_edges=int(float(r.get("n_nonzero_edges", 0) or 0)),
                )
            )
    if not out:
        raise ValueError(f"No status=ok rows in metrics csv: {path}")
    return out


def _read_manifest(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = {(r.get("sample_id") or "").strip(): r for r in reader}
    return fields, rows


def _pick_low_mid_high(rows: list[CaseRow], key: str) -> list[tuple[str, CaseRow]]:
    rows_sorted = sorted(rows, key=lambda x: getattr(x, key))
    picked = [
        ("low", rows_sorted[0]),
        ("mid", rows_sorted[len(rows_sorted) // 2]),
        ("high", rows_sorted[-1]),
    ]
    # de-dup if collisions
    seen = set()
    uniq = []
    for tag, row in picked:
        if row.sample_id in seen:
            continue
        seen.add(row.sample_id)
        uniq.append((tag, row))
    return uniq


def _write_selected_cases_csv(path: Path, picked: list[tuple[str, CaseRow]], key: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank_tag",
                "sample_id",
                "selected_by",
                "replacement_score",
                "completeness_score",
                "n_nodes_total",
                "n_nonzero_edges",
            ],
        )
        writer.writeheader()
        for tag, row in picked:
            writer.writerow(
                {
                    "rank_tag": tag,
                    "sample_id": row.sample_id,
                    "selected_by": key,
                    "replacement_score": row.replacement_score,
                    "completeness_score": row.completeness_score,
                    "n_nodes_total": row.n_nodes_total,
                    "n_nonzero_edges": row.n_nonzero_edges,
                }
            )


def _write_selected_manifest(
    path: Path,
    manifest_fields: list[str],
    manifest_by_id: dict[str, dict[str, str]],
    picked: list[tuple[str, CaseRow]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for _, row in picked:
            if row.sample_id in manifest_by_id:
                writer.writerow(manifest_by_id[row.sample_id])


def _topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    k = min(k, values.numel())
    _, idx = torch.topk(values.abs(), k=k)
    return idx


def _analyze_single_graph(
    graph_path: Path,
    topk_features: int,
    topk_errors: int,
    topk_tokens: int,
    tok_cache: dict[str, AutoTokenizer],
) -> dict:
    g = Graph.from_pt(str(graph_path), map_location="cpu")
    n_total = int(g.adjacency_matrix.shape[0])
    n_logits = int(len(g.logit_tokens))
    n_tokens = int(len(g.input_tokens))
    n_features = int(len(g.selected_features))
    n_pos = int(g.n_pos)
    n_layers = int(g.cfg.n_layers)
    n_errors = n_layers * n_pos

    error_start = n_features
    error_end = error_start + n_errors
    token_end = error_end + n_tokens
    target_logit_row = n_total - n_logits  # first/top logit
    row = g.adjacency_matrix[target_logit_row].detach().float().cpu()

    tokenizer_name = g.cfg.tokenizer_name
    if tokenizer_name not in tok_cache:
        tok_cache[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
    tok = tok_cache[tokenizer_name]

    # Feature contributors
    feat_scores = row[:n_features]
    feat_idx = _topk_indices(feat_scores, topk_features)
    top_features = []
    for local_idx in feat_idx.tolist():
        active_idx = int(g.selected_features[local_idx])
        layer, pos, feat_id = g.active_features[active_idx].tolist()
        top_features.append(
            {
                "local_feature_idx": local_idx,
                "layer": int(layer),
                "pos": int(pos),
                "feature_id": int(feat_id),
                "weight_to_target_logit": float(feat_scores[local_idx]),
                "activation_value": float(g.activation_values[active_idx]),
            }
        )

    # Error contributors
    err_scores = row[error_start:error_end]
    err_idx = _topk_indices(err_scores, topk_errors)
    top_errors = []
    for rel in err_idx.tolist():
        layer, pos = divmod(rel, n_pos)
        top_errors.append(
            {
                "error_flat_idx": int(rel),
                "layer": int(layer),
                "pos": int(pos),
                "weight_to_target_logit": float(err_scores[rel]),
            }
        )

    # Token contributors
    tok_scores = row[error_end:token_end]
    tok_idx = _topk_indices(tok_scores, topk_tokens)
    top_tokens = []
    for pos in tok_idx.tolist():
        token_id = int(g.input_tokens[pos])
        top_tokens.append(
            {
                "pos": int(pos),
                "token_id": token_id,
                "token_str": tok.decode([token_id]),
                "weight_to_target_logit": float(tok_scores[pos]),
            }
        )

    return {
        "target_logit_token_id": int(g.logit_tokens[0]),
        "target_logit_token_str": tok.decode([int(g.logit_tokens[0])]),
        "target_logit_prob": float(g.logit_probabilities[0]),
        "top_features": top_features,
        "top_errors": top_errors,
        "top_tokens": top_tokens,
    }


def _write_table_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick 3 representative cases and generate text analysis.")
    parser.add_argument("--metrics-csv", required=True, help="Path to metrics_stream csv.")
    parser.add_argument("--manifest-csv", required=True, help="Path to source sample manifest csv.")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports.")
    parser.add_argument(
        "--select-by",
        default="replacement_score",
        choices=["replacement_score", "completeness_score"],
        help="Metric used to pick low/mid/high representative cases.",
    )
    parser.add_argument(
        "--pt-dir",
        default="",
        help="Directory with sample_id.pt files. If missing, report will include only metadata.",
    )
    parser.add_argument("--topk-features", type=int, default=20)
    parser.add_argument("--topk-errors", type=int, default=10)
    parser.add_argument("--topk-tokens", type=int, default=10)
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_metrics(metrics_csv)
    manifest_fields, manifest_by_id = _read_manifest(manifest_csv)
    picked = _pick_low_mid_high(rows, args.select_by)

    selected_cases_csv = out_dir / "selected_cases.csv"
    selected_manifest_csv = out_dir / "selected_manifest.csv"
    _write_selected_cases_csv(selected_cases_csv, picked, args.select_by)
    _write_selected_manifest(selected_manifest_csv, manifest_fields, manifest_by_id, picked)

    report_lines = []
    report_lines.append("# Three-Case Analysis")
    report_lines.append("")
    report_lines.append(f"- Selected by: `{args.select_by}`")
    report_lines.append(f"- Metrics CSV: `{metrics_csv}`")
    report_lines.append(f"- Manifest CSV: `{manifest_csv}`")
    report_lines.append("")

    pt_dir = Path(args.pt_dir).expanduser().resolve() if args.pt_dir else None
    tok_cache: dict[str, AutoTokenizer] = {}

    for tag, r in picked:
        report_lines.append(f"## Case `{tag}` - `{r.sample_id}`")
        report_lines.append(
            f"- replacement={r.replacement_score:.6f}, completeness={r.completeness_score:.6f}, "
            f"nodes={r.n_nodes_total}, edges={r.n_nonzero_edges}"
        )
        m = manifest_by_id.get(r.sample_id, {})
        if m:
            report_lines.append(f"- question: {m.get('question','')}")
            report_lines.append(f"- image_path: {m.get('image_path','')}")

        if not pt_dir:
            report_lines.append("- pt analysis: skipped (`--pt-dir` not provided)")
            report_lines.append("")
            continue

        pt_path = pt_dir / f"{r.sample_id}.pt"
        if not pt_path.exists():
            report_lines.append(f"- pt analysis: missing file `{pt_path}`")
            report_lines.append("")
            continue

        analysis = _analyze_single_graph(
            pt_path,
            topk_features=args.topk_features,
            topk_errors=args.topk_errors,
            topk_tokens=args.topk_tokens,
            tok_cache=tok_cache,
        )
        report_lines.append(
            f"- target_logit: `{analysis['target_logit_token_str']}` "
            f"(id={analysis['target_logit_token_id']}, prob={analysis['target_logit_prob']:.4f})"
        )

        # per-case csv outputs
        feature_csv = out_dir / f"{r.sample_id}_top_features.csv"
        error_csv = out_dir / f"{r.sample_id}_top_errors.csv"
        token_csv = out_dir / f"{r.sample_id}_top_tokens.csv"
        _write_table_csv(feature_csv, analysis["top_features"])
        _write_table_csv(error_csv, analysis["top_errors"])
        _write_table_csv(token_csv, analysis["top_tokens"])

        report_lines.append(f"- top features csv: `{feature_csv.name}`")
        report_lines.append(f"- top errors csv: `{error_csv.name}`")
        report_lines.append(f"- top tokens csv: `{token_csv.name}`")
        report_lines.append("")

    report_path = out_dir / "three_case_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote: {selected_cases_csv}")
    print(f"[ok] wrote: {selected_manifest_csv}")
    print(f"[ok] wrote: {report_path}")
    if pt_dir:
        print(f"[info] pt_dir: {pt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
