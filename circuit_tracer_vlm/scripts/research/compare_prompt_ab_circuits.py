#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
import types
from collections import defaultdict
from pathlib import Path

import torch


def _ensure_transformer_lens_stub_for_torch_load() -> None:
    # Graph .pt stores cfg object typed as HookedTransformerConfig.
    # For analysis scripts, we only need cfg.n_layers, so a tiny stub is enough.
    if "transformer_lens.HookedTransformerConfig" in sys.modules:
        return
    pkg = types.ModuleType("transformer_lens")
    pkg.__path__ = []
    sub = types.ModuleType("transformer_lens.HookedTransformerConfig")

    class HookedTransformerConfig:  # noqa: N801
        pass

    sub.HookedTransformerConfig = HookedTransformerConfig
    sys.modules["transformer_lens"] = pkg
    sys.modules["transformer_lens.HookedTransformerConfig"] = sub


def _ensure_transformer_lens_with_vl_if_needed() -> None:
    # Optional: for importing circuit_tracer.graph (replacement/completeness exact impl).
    script_path = Path(__file__).resolve()
    candidate_paths = [
        script_path.parents[3] / "third_party" / "TransformerLens",
        script_path.parents[2] / "third_party" / "TransformerLens",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
    for name in list(sys.modules.keys()):
        if name == "transformer_lens" or name.startswith("transformer_lens."):
            del sys.modules[name]
    lens = importlib.import_module("transformer_lens")
    if not hasattr(lens, "HookedVLTransformer"):
        raise ImportError(
            "transformer_lens does not provide HookedVLTransformer. "
            "Use vendored third_party/TransformerLens."
        )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return len(a & b) / u


def _normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix.abs()
    denom = m.sum(dim=1, keepdim=True).clamp(min=1e-10)
    return m / denom


def _compute_influence(a_norm: torch.Tensor, logit_weights: torch.Tensor, max_iter: int = 1000) -> torch.Tensor:
    current = logit_weights @ a_norm
    influence = current.clone()
    n_iter = 0
    while current.any():
        if n_iter >= max_iter:
            raise RuntimeError(f"influence did not converge after {n_iter} iterations")
        current = current @ a_norm
        influence = influence + current
        n_iter += 1
    return influence


def _compute_replacement_completeness(d: dict) -> tuple[float, float]:
    a = d["adjacency_matrix"].float().cpu()
    logit_probs = d["logit_probabilities"].float().cpu()
    n_logits = int(len(d["logit_tokens"]))
    n_tokens = int(len(d["input_tokens"]))
    n_layers = int(d["cfg"].n_layers)
    n_features = int(len(d["selected_features"]))

    error_start = n_features
    error_end = error_start + n_tokens * n_layers
    token_end = error_end + n_tokens

    logit_weights = torch.zeros(a.shape[0], dtype=a.dtype)
    logit_weights[-n_logits:] = logit_probs
    a_norm = _normalize_matrix(a)
    node_influence = _compute_influence(a_norm, logit_weights)
    token_influence = node_influence[error_end:token_end].sum()
    error_influence = node_influence[error_start:error_end].sum()
    replacement = _safe_div(float(token_influence), float(token_influence + error_influence))

    non_error_frac = 1 - a_norm[:, error_start:error_end].sum(dim=-1)
    output_influence = node_influence + logit_weights
    completeness = _safe_div(
        float((non_error_frac * output_influence).sum()),
        float(output_influence.sum()),
    )
    return replacement, completeness


def _load_pt(path: Path) -> dict:
    _ensure_transformer_lens_stub_for_torch_load()
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_index(d: dict, target_logit_rank: int) -> dict:
    cfg = d["cfg"]
    input_tokens = d["input_tokens"].cpu()
    active_features = d["active_features"].cpu()
    selected_features = d["selected_features"].cpu()
    logit_tokens = d["logit_tokens"].cpu()

    n_layers = int(cfg.n_layers)
    n_pos = int(len(input_tokens))
    n_features = int(len(selected_features))
    n_errors = n_layers * n_pos
    n_tokens = n_pos
    n_logits = int(len(logit_tokens))

    if target_logit_rank < 0 or target_logit_rank >= n_logits:
        raise ValueError(f"target_logit_rank out of range: {target_logit_rank}")

    error_start = n_features
    error_end = error_start + n_errors
    token_start = error_end
    token_end = token_start + n_tokens
    logit_start = token_end
    target_row = logit_start + target_logit_rank

    node_ids: list[str] = [""] * (logit_start + n_logits)
    stage: list[int] = [0] * (logit_start + n_logits)

    for local_idx in range(n_features):
        active_idx = int(selected_features[local_idx])
        layer, pos, feat = [int(x) for x in active_features[active_idx].tolist()]
        node_ids[local_idx] = f"F:L{layer}:P{pos}:ID{feat}"
        stage[local_idx] = layer

    for rel in range(n_errors):
        idx = error_start + rel
        layer = rel // n_pos
        pos = rel % n_pos
        node_ids[idx] = f"E:L{layer}:P{pos}"
        stage[idx] = layer

    for pos in range(n_tokens):
        idx = token_start + pos
        tok_id = int(input_tokens[pos])
        node_ids[idx] = f"T:P{pos}:ID{tok_id}"
        stage[idx] = -1

    for r in range(n_logits):
        idx = logit_start + r
        tok_id = int(logit_tokens[r])
        node_ids[idx] = f"L:R{r}:ID{tok_id}"
        stage[idx] = n_layers + 1

    return {
        "n_layers": n_layers,
        "n_pos": n_pos,
        "n_features": n_features,
        "n_errors": n_errors,
        "n_tokens": n_tokens,
        "n_logits": n_logits,
        "error_start": error_start,
        "error_end": error_end,
        "token_start": token_start,
        "token_end": token_end,
        "logit_start": logit_start,
        "target_row": target_row,
        "target_token_id": int(logit_tokens[target_logit_rank]),
        "node_ids": node_ids,
        "stage": stage,
    }


def _trace_strong_backpath(
    a: torch.Tensor,
    stage: list[int],
    target_row: int,
    parents_per_node: int,
    max_depth: int,
    min_abs_weight: float,
) -> tuple[list[tuple[int, int, float]], dict[int, int]]:
    # Returns edges as (src, dst, w) and node_depth where target depth is 0.
    edges: list[tuple[int, int, float]] = []
    depth_map: dict[int, int] = {target_row: 0}
    frontier = {target_row}

    for depth in range(1, max_depth + 1):
        next_frontier: set[int] = set()
        for dst in frontier:
            incoming = a[dst]
            nz = torch.nonzero(incoming != 0, as_tuple=False).flatten().tolist()
            cands: list[tuple[float, int, float]] = []
            for src in nz:
                if stage[src] >= stage[dst]:
                    continue
                w = float(incoming[src])
                aw = abs(w)
                if aw < min_abs_weight:
                    continue
                cands.append((aw, src, w))
            cands.sort(key=lambda x: x[0], reverse=True)
            for _, src, w in cands[:parents_per_node]:
                edges.append((src, dst, w))
                next_frontier.add(src)
                prev = depth_map.get(src, -1)
                if depth > prev:
                    depth_map[src] = depth
        if not next_frontier:
            break
        frontier = next_frontier
    return edges, depth_map


def _compute_run_metrics(
    pt_path: Path,
    target_logit_rank: int,
    parents_per_node: int,
    max_depth: int,
    min_abs_weight: float,
    include_graph_scores: bool,
) -> dict:
    d = _load_pt(pt_path)
    a = d["adjacency_matrix"].float().cpu()
    idx = _build_index(d=d, target_logit_rank=target_logit_rank)
    target_row = idx["target_row"]
    node_ids = idx["node_ids"]
    stage = idx["stage"]

    incoming = a[target_row]
    incoming_abs = incoming.abs()
    total_in = float(incoming_abs.sum().item())
    top_vals = torch.sort(incoming_abs, descending=True).values
    top1 = float(top_vals[0].item()) if len(top_vals) >= 1 else 0.0
    top3 = float(top_vals[:3].sum().item()) if len(top_vals) >= 3 else float(top_vals.sum().item())
    top10 = float(top_vals[:10].sum().item()) if len(top_vals) >= 10 else float(top_vals.sum().item())

    error_abs = float(incoming_abs[idx["error_start"] : idx["error_end"]].sum().item())
    feature_abs = float(incoming_abs[: idx["n_features"]].sum().item())
    token_abs = float(incoming_abs[idx["token_start"] : idx["token_end"]].sum().item())

    traced_edges, depth_map = _trace_strong_backpath(
        a=a,
        stage=stage,
        target_row=target_row,
        parents_per_node=parents_per_node,
        max_depth=max_depth,
        min_abs_weight=min_abs_weight,
    )
    traced_nodes_idx = {target_row}
    for src, dst, _ in traced_edges:
        traced_nodes_idx.add(src)
        traced_nodes_idx.add(dst)
    traced_node_ids = {node_ids[i] for i in traced_nodes_idx}
    traced_edge_ids = {f"{node_ids[src]}->{node_ids[dst]}" for src, dst, _ in traced_edges}
    traced_max_depth = max(depth_map.values()) if depth_map else 0

    replacement_score = math.nan
    completeness_score = math.nan
    if include_graph_scores:
        replacement_score, completeness_score = _compute_replacement_completeness(d)

    return {
        "sample_id": pt_path.stem,
        "target_token_id": idx["target_token_id"],
        "n_total_nodes": int(a.shape[0]),
        "n_feature_nodes": int(idx["n_features"]),
        "n_nonzero_edges": int((a != 0).sum().item()),
        "target_total_in_abs": total_in,
        "target_top1_concentration": _safe_div(top1, total_in),
        "target_top3_concentration": _safe_div(top3, total_in),
        "target_top10_concentration": _safe_div(top10, total_in),
        "target_error_ratio": _safe_div(error_abs, total_in),
        "target_feature_ratio": _safe_div(feature_abs, total_in),
        "target_token_ratio": _safe_div(token_abs, total_in),
        "traced_nodes": len(traced_node_ids),
        "traced_edges": len(traced_edge_ids),
        "traced_max_depth": int(traced_max_depth),
        "replacement_score": replacement_score,
        "completeness_score": completeness_score,
        "_traced_node_set": traced_node_ids,
        "_traced_edge_set": traced_edge_ids,
    }


def _read_bucket_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, str] = {}
    for r in rows:
        sid = (r.get("sample_id") or "").strip()
        if not sid:
            continue
        if r.get("bucket"):
            out[sid] = (r.get("bucket") or "").strip()
            continue
        a = (r.get("a_correct") or r.get("prompt_a_correct") or "").strip().lower()
        b = (r.get("b_correct") or r.get("prompt_b_correct") or "").strip().lower()
        if a in {"0", "1", "true", "false"} and b in {"0", "1", "true", "false"}:
            av = a in {"1", "true"}
            bv = b in {"1", "true"}
            out[sid] = f"A{int(av)}_B{int(bv)}"
    return out


def _mean(vals: list[float]) -> float:
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare circuit metrics for prompt A vs prompt B (or run1 vs run2) "
            "on matched sample_id .pt files."
        )
    )
    parser.add_argument("--pt-dir-a", required=True, help="Directory of .pt files for run A")
    parser.add_argument("--pt-dir-b", required=True, help="Directory of .pt files for run B")
    parser.add_argument("--out-sample-csv", required=True, help="Per-sample comparison csv")
    parser.add_argument("--out-summary-csv", required=True, help="Bucket-level summary csv")
    parser.add_argument("--bucket-csv", default="", help="Optional csv with sample_id and bucket")
    parser.add_argument("--target-logit-rank", type=int, default=0)
    parser.add_argument("--parents-per-node", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=48)
    parser.add_argument("--min-abs-weight", type=float, default=0.0)
    parser.add_argument(
        "--skip-graph-scores",
        action="store_true",
        help="Skip replacement/completeness (faster, less memory).",
    )
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    pt_dir_a = Path(args.pt_dir_a).expanduser().resolve()
    pt_dir_b = Path(args.pt_dir_b).expanduser().resolve()
    out_sample_csv = Path(args.out_sample_csv).expanduser().resolve()
    out_summary_csv = Path(args.out_summary_csv).expanduser().resolve()

    bucket_map: dict[str, str] = {}
    if args.bucket_csv:
        bucket_map = _read_bucket_map(Path(args.bucket_csv).expanduser().resolve())

    files_a = {p.stem: p for p in pt_dir_a.glob("*.pt")}
    files_b = {p.stem: p for p in pt_dir_b.glob("*.pt")}
    common_ids = sorted(set(files_a.keys()) & set(files_b.keys()))
    if not common_ids:
        raise ValueError("no common sample_id .pt files between A and B")

    include_graph_scores = not args.skip_graph_scores
    rows: list[dict] = []
    total = len(common_ids)
    for i, sid in enumerate(common_ids, start=1):
        a_metrics = _compute_run_metrics(
            pt_path=files_a[sid],
            target_logit_rank=args.target_logit_rank,
            parents_per_node=args.parents_per_node,
            max_depth=args.max_depth,
            min_abs_weight=args.min_abs_weight,
            include_graph_scores=include_graph_scores,
        )
        b_metrics = _compute_run_metrics(
            pt_path=files_b[sid],
            target_logit_rank=args.target_logit_rank,
            parents_per_node=args.parents_per_node,
            max_depth=args.max_depth,
            min_abs_weight=args.min_abs_weight,
            include_graph_scores=include_graph_scores,
        )

        row = {
            "sample_id": sid,
            "bucket": bucket_map.get(sid, "all"),
            "node_overlap_jaccard": _jaccard(a_metrics["_traced_node_set"], b_metrics["_traced_node_set"]),
            "edge_overlap_jaccard": _jaccard(a_metrics["_traced_edge_set"], b_metrics["_traced_edge_set"]),
        }

        metric_keys = [
            "target_token_id",
            "n_total_nodes",
            "n_feature_nodes",
            "n_nonzero_edges",
            "target_total_in_abs",
            "target_top1_concentration",
            "target_top3_concentration",
            "target_top10_concentration",
            "target_error_ratio",
            "target_feature_ratio",
            "target_token_ratio",
            "traced_nodes",
            "traced_edges",
            "traced_max_depth",
            "replacement_score",
            "completeness_score",
        ]
        for k in metric_keys:
            av = a_metrics[k]
            bv = b_metrics[k]
            row[f"a_{k}"] = av
            row[f"b_{k}"] = bv
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                row[f"delta_{k}"] = float(av) - float(bv)
            else:
                row[f"delta_{k}"] = ""
        rows.append(row)

        if i % max(1, args.log_every) == 0 or i == total:
            print(f"[progress] {i}/{total} ({i/total*100:.1f}%)")

    # write sample-level csv
    sample_fields = list(rows[0].keys()) if rows else ["sample_id"]
    _write_csv(out_sample_csv, rows, sample_fields)

    # bucket-level summary (means)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    summary_rows: list[dict] = []
    summary_metrics = [
        "node_overlap_jaccard",
        "edge_overlap_jaccard",
        "a_target_top1_concentration",
        "b_target_top1_concentration",
        "delta_target_top1_concentration",
        "a_target_top3_concentration",
        "b_target_top3_concentration",
        "delta_target_top3_concentration",
        "a_target_top10_concentration",
        "b_target_top10_concentration",
        "delta_target_top10_concentration",
        "a_target_error_ratio",
        "b_target_error_ratio",
        "delta_target_error_ratio",
        "a_traced_max_depth",
        "b_traced_max_depth",
        "delta_traced_max_depth",
        "a_traced_nodes",
        "b_traced_nodes",
        "delta_traced_nodes",
        "a_traced_edges",
        "b_traced_edges",
        "delta_traced_edges",
        "a_replacement_score",
        "b_replacement_score",
        "delta_replacement_score",
        "a_completeness_score",
        "b_completeness_score",
        "delta_completeness_score",
    ]
    for bucket, bucket_rows in sorted(by_bucket.items(), key=lambda x: x[0]):
        out = {"bucket": bucket, "count": len(bucket_rows)}
        for m in summary_metrics:
            vals = [float(r[m]) for r in bucket_rows if isinstance(r.get(m), (int, float))]
            out[m] = _mean(vals)
        summary_rows.append(out)

    # overall line
    all_row = {"bucket": "__all__", "count": len(rows)}
    for m in summary_metrics:
        vals = [float(r[m]) for r in rows if isinstance(r.get(m), (int, float))]
        all_row[m] = _mean(vals)
    summary_rows.append(all_row)

    summary_fields = list(summary_rows[0].keys()) if summary_rows else ["bucket", "count"]
    _write_csv(out_summary_csv, summary_rows, summary_fields)

    print(f"[done] sample csv: {out_sample_csv}")
    print(f"[done] summary csv: {out_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
