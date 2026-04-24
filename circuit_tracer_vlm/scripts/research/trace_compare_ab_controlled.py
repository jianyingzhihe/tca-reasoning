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
            "Use vendored third_party/TransformerLens fork."
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


def _load_pt(path: Path) -> dict:
    _ensure_transformer_lens_stub_for_torch_load()
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_index(d: dict, target_logit_rank: int) -> dict:
    input_tokens = d["input_tokens"].cpu()
    active_features = d["active_features"].cpu()
    selected_features = d["selected_features"].cpu()
    logit_tokens = d["logit_tokens"].cpu()
    adjacency_matrix = d["adjacency_matrix"]

    n_pos = int(len(input_tokens))
    n_features = int(len(selected_features))
    n_tokens = n_pos
    n_logits = int(len(logit_tokens))
    total_nodes = int(adjacency_matrix.shape[0])
    n_errors = total_nodes - n_features - n_tokens - n_logits
    if n_errors < 0:
        raise ValueError(
            f"invalid graph layout: total_nodes={total_nodes} n_features={n_features} "
            f"n_tokens={n_tokens} n_logits={n_logits}"
        )
    if n_pos <= 0:
        raise ValueError("graph has no input tokens")
    if n_errors % n_pos != 0:
        raise ValueError(
            f"cannot infer error layout: n_errors={n_errors} is not divisible by n_pos={n_pos}"
        )
    n_layers = n_errors // n_pos

    if target_logit_rank < 0 or target_logit_rank >= n_logits:
        raise ValueError(f"target_logit_rank out of range: {target_logit_rank}")

    error_start = n_features
    error_end = error_start + n_errors
    token_start = error_end
    token_end = token_start + n_tokens
    logit_start = token_end
    target_row = logit_start + target_logit_rank

    node_ids: list[str] = [""] * total_nodes
    stage: list[int] = [0] * total_nodes
    node_meta: list[dict] = [None] * total_nodes  # type: ignore

    for local_idx in range(n_features):
        active_idx = int(selected_features[local_idx])
        layer, pos, feat = [int(x) for x in active_features[active_idx].tolist()]
        node_ids[local_idx] = f"F:L{layer}:P{pos}:ID{feat}"
        stage[local_idx] = layer
        node_meta[local_idx] = {
            "node_type": "feature",
            "layer": layer,
            "pos": pos,
            "feature_id": feat,
            "token_id": "",
        }

    for rel in range(n_errors):
        idx = error_start + rel
        layer = rel // n_pos
        pos = rel % n_pos
        node_ids[idx] = f"E:L{layer}:P{pos}"
        stage[idx] = layer
        node_meta[idx] = {
            "node_type": "error",
            "layer": layer,
            "pos": pos,
            "feature_id": "",
            "token_id": "",
        }

    for pos in range(n_tokens):
        idx = token_start + pos
        tok_id = int(input_tokens[pos])
        node_ids[idx] = f"T:P{pos}:ID{tok_id}"
        stage[idx] = -1
        node_meta[idx] = {
            "node_type": "token",
            "layer": -1,
            "pos": pos,
            "feature_id": "",
            "token_id": tok_id,
        }

    for r in range(n_logits):
        idx = logit_start + r
        tok_id = int(logit_tokens[r])
        node_ids[idx] = f"L:R{r}:ID{tok_id}"
        stage[idx] = n_layers + 1
        node_meta[idx] = {
            "node_type": "logit",
            "layer": n_layers + 1,
            "pos": r,
            "feature_id": "",
            "token_id": tok_id,
        }

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
        "node_meta": node_meta,
    }


def _controlled_backtrace(
    a: torch.Tensor,
    stage: list[int],
    target_row: int,
    topk_per_node: int,
    beam_per_depth: int,
    coverage: float,
    max_depth: int,
    min_abs_weight: float,
) -> tuple[list[dict], dict[int, int], dict[int, float]]:
    frontier = {target_row}
    depth_map: dict[int, int] = {target_row: 0}
    path_mass: dict[int, float] = {target_row: 1.0}
    edges_out: list[dict] = []

    for depth in range(1, max_depth + 1):
        candidates: list[dict] = []
        for dst in frontier:
            incoming = a[dst]
            nz = torch.nonzero(incoming != 0, as_tuple=False).flatten().tolist()
            if not nz:
                continue
            dst_total = float(incoming.abs().sum().item())
            if dst_total <= 0:
                continue

            node_cands: list[dict] = []
            for src in nz:
                if stage[src] >= stage[dst]:
                    continue
                w = float(incoming[src].item())
                aw = abs(w)
                if aw < min_abs_weight:
                    continue
                local_ratio = _safe_div(aw, dst_total)
                pmass = path_mass.get(dst, 0.0) * local_ratio
                node_cands.append(
                    {
                        "src": src,
                        "dst": dst,
                        "weight": w,
                        "abs_weight": aw,
                        "local_ratio": local_ratio,
                        "dst_path_mass": path_mass.get(dst, 0.0),
                        "path_mass": pmass,
                    }
                )

            if not node_cands:
                continue
            node_cands.sort(key=lambda x: x["abs_weight"], reverse=True)
            candidates.extend(node_cands[: max(1, topk_per_node)])

        if not candidates:
            break

        # Dedupe same (src,dst) by best path_mass.
        edge_best: dict[tuple[int, int], dict] = {}
        for c in candidates:
            k = (c["src"], c["dst"])
            prev = edge_best.get(k)
            if prev is None or c["path_mass"] > prev["path_mass"]:
                edge_best[k] = c
        candidates = list(edge_best.values())
        candidates.sort(key=lambda x: (x["path_mass"], x["abs_weight"]), reverse=True)

        total_mass = sum(c["path_mass"] for c in candidates)
        keep_n = min(len(candidates), max(1, beam_per_depth))
        if total_mass > 0 and coverage > 0:
            acc = 0.0
            keep_cov = 0
            for c in candidates:
                acc += c["path_mass"]
                keep_cov += 1
                if _safe_div(acc, total_mass) >= coverage:
                    break
            keep_n = min(keep_n, max(1, keep_cov))

        kept = candidates[:keep_n]
        next_frontier: set[int] = set()
        next_mass_best: dict[int, float] = {}
        for c in kept:
            src = int(c["src"])
            dst = int(c["dst"])
            edges_out.append(
                {
                    "depth": depth,
                    "src": src,
                    "dst": dst,
                    "weight": float(c["weight"]),
                    "abs_weight": float(c["abs_weight"]),
                    "local_ratio": float(c["local_ratio"]),
                    "dst_path_mass": float(c["dst_path_mass"]),
                    "path_mass": float(c["path_mass"]),
                }
            )
            next_frontier.add(src)
            depth_map[src] = max(depth_map.get(src, 0), depth)
            next_mass_best[src] = max(next_mass_best.get(src, 0.0), float(c["path_mass"]))

        if not next_frontier:
            break
        frontier = next_frontier
        for n, m in next_mass_best.items():
            path_mass[n] = max(path_mass.get(n, 0.0), m)

    return edges_out, depth_map, path_mass


def _compute_run(
    pt_path: Path,
    run_name: str,
    sample_id: str,
    bucket: str,
    target_logit_rank: int,
    topk_per_node: int,
    beam_per_depth: int,
    coverage: float,
    max_depth: int,
    min_abs_weight: float,
) -> tuple[dict, list[dict], list[dict], set[str], set[str]]:
    d = _load_pt(pt_path)
    a = d["adjacency_matrix"].float().cpu()
    idx = _build_index(d=d, target_logit_rank=target_logit_rank)
    target_row = idx["target_row"]
    node_ids = idx["node_ids"]
    stage = idx["stage"]
    node_meta = idx["node_meta"]

    edges_raw, depth_map, path_mass = _controlled_backtrace(
        a=a,
        stage=stage,
        target_row=target_row,
        topk_per_node=topk_per_node,
        beam_per_depth=beam_per_depth,
        coverage=coverage,
        max_depth=max_depth,
        min_abs_weight=min_abs_weight,
    )

    traced_nodes_idx = {target_row}
    for e in edges_raw:
        traced_nodes_idx.add(int(e["src"]))
        traced_nodes_idx.add(int(e["dst"]))

    edge_rows: list[dict] = []
    for e in edges_raw:
        src = int(e["src"])
        dst = int(e["dst"])
        edge_rows.append(
            {
                "sample_id": sample_id,
                "bucket": bucket,
                "run": run_name,
                "depth": int(e["depth"]),
                "src_idx": src,
                "dst_idx": dst,
                "src_node": node_ids[src],
                "dst_node": node_ids[dst],
                "src_layer": int(stage[src]),
                "dst_layer": int(stage[dst]),
                "weight": float(e["weight"]),
                "abs_weight": float(e["abs_weight"]),
                "local_ratio": float(e["local_ratio"]),
                "dst_path_mass": float(e["dst_path_mass"]),
                "path_mass": float(e["path_mass"]),
            }
        )

    node_rows: list[dict] = []
    for n in sorted(traced_nodes_idx):
        meta = node_meta[n]
        node_rows.append(
            {
                "sample_id": sample_id,
                "bucket": bucket,
                "run": run_name,
                "node_idx": int(n),
                "node_id": node_ids[n],
                "node_type": meta["node_type"],
                "layer": int(meta["layer"]),
                "pos": int(meta["pos"]) if meta["pos"] != "" else "",
                "feature_id": meta["feature_id"],
                "token_id": meta["token_id"],
                "depth_from_target": int(depth_map.get(n, 0)),
                "path_mass_best": float(path_mass.get(n, 0.0)),
            }
        )

    incoming = a[target_row]
    incoming_abs = incoming.abs()
    total_in = float(incoming_abs.sum().item())
    error_abs = float(incoming_abs[idx["error_start"] : idx["error_end"]].sum().item())
    feature_abs = float(incoming_abs[: idx["n_features"]].sum().item())
    token_abs = float(incoming_abs[idx["token_start"] : idx["token_end"]].sum().item())

    top_vals = torch.sort(incoming_abs, descending=True).values
    top1 = float(top_vals[0].item()) if len(top_vals) >= 1 else 0.0
    top3 = float(top_vals[:3].sum().item()) if len(top_vals) >= 3 else float(top_vals.sum().item())
    top10 = float(top_vals[:10].sum().item()) if len(top_vals) >= 10 else float(top_vals.sum().item())

    summary = {
        "sample_id": sample_id,
        "bucket": bucket,
        "run": run_name,
        "target_token_id": int(idx["target_token_id"]),
        "n_layers": int(idx["n_layers"]),
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
        "traced_nodes": int(len(traced_nodes_idx)),
        "traced_edges": int(len(edges_raw)),
        "traced_max_depth": int(max(depth_map.values()) if depth_map else 0),
        "traced_total_path_mass": float(sum(e["path_mass"] for e in edges_raw)),
        "traced_min_abs_weight": float(min((e["abs_weight"] for e in edges_raw), default=0.0)),
        "traced_max_abs_weight": float(max((e["abs_weight"] for e in edges_raw), default=0.0)),
    }

    node_set = {node_ids[i] for i in traced_nodes_idx}
    edge_set = {f"{node_ids[int(e['src'])]}->{node_ids[int(e['dst'])]}" for e in edges_raw}
    return summary, node_rows, edge_rows, node_set, edge_set


def _read_bucket_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sample_by_bucket(rows: list[dict], buckets: list[str], per_bucket: int) -> list[dict]:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        b = (r.get("bucket") or "").strip()
        sid = (r.get("sample_id") or "").strip()
        if not b or not sid:
            continue
        by_bucket[b].append(r)

    selected: list[dict] = []
    for b in buckets:
        part = by_bucket.get(b, [])
        part_sorted = sorted(part, key=lambda x: (x.get("sample_id") or ""))
        selected.extend(part_sorted[: max(0, per_bucket)])
    return selected


def _mean(vals: list[float]) -> float:
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled A/B backtrace export with topk-per-node + layer beam + coverage. "
            "Designed for detailed yet disk-safe circuit analysis."
        )
    )
    parser.add_argument("--pt-dir-a", required=True)
    parser.add_argument("--pt-dir-b", required=True)
    parser.add_argument("--bucket-csv", required=True, help="CSV with sample_id,bucket")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--buckets", default="A0_B1,A1_B1,A0_B0,A1_B0")
    parser.add_argument("--per-bucket", type=int, default=60)
    parser.add_argument("--target-logit-rank", type=int, default=0)
    parser.add_argument("--topk-per-node", type=int, default=3)
    parser.add_argument("--beam-per-depth", type=int, default=64)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--min-abs-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    pt_dir_a = Path(args.pt_dir_a).expanduser().resolve()
    pt_dir_b = Path(args.pt_dir_b).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_rows = _read_bucket_rows(Path(args.bucket_csv).expanduser().resolve())
    buckets = [x.strip() for x in args.buckets.split(",") if x.strip()]
    selected_rows = _sample_by_bucket(bucket_rows, buckets, args.per_bucket)
    if not selected_rows:
        raise ValueError("no selected samples from bucket csv")

    files_a = {p.stem: p for p in pt_dir_a.glob("*.pt")}
    files_b = {p.stem: p for p in pt_dir_b.glob("*.pt")}

    selected_with_files: list[dict] = []
    dropped = 0
    for r in selected_rows:
        sid = (r.get("sample_id") or "").strip()
        if sid in files_a and sid in files_b:
            selected_with_files.append(r)
        else:
            dropped += 1

    if not selected_with_files:
        raise ValueError("selected samples have no matched A/B .pt files")

    _write_csv(
        out_dir / "selected_samples.csv",
        selected_with_files,
        fieldnames=list(selected_with_files[0].keys()),
    )

    summary_rows_a: list[dict] = []
    summary_rows_b: list[dict] = []
    sample_compare_rows: list[dict] = []
    node_rows: list[dict] = []
    edge_rows: list[dict] = []

    total = len(selected_with_files)
    for i, r in enumerate(selected_with_files, start=1):
        sid = (r.get("sample_id") or "").strip()
        bucket = (r.get("bucket") or "").strip()

        s_a, n_a, e_a, set_n_a, set_e_a = _compute_run(
            pt_path=files_a[sid],
            run_name="A",
            sample_id=sid,
            bucket=bucket,
            target_logit_rank=args.target_logit_rank,
            topk_per_node=args.topk_per_node,
            beam_per_depth=args.beam_per_depth,
            coverage=args.coverage,
            max_depth=args.max_depth,
            min_abs_weight=args.min_abs_weight,
        )
        s_b, n_b, e_b, set_n_b, set_e_b = _compute_run(
            pt_path=files_b[sid],
            run_name="B",
            sample_id=sid,
            bucket=bucket,
            target_logit_rank=args.target_logit_rank,
            topk_per_node=args.topk_per_node,
            beam_per_depth=args.beam_per_depth,
            coverage=args.coverage,
            max_depth=args.max_depth,
            min_abs_weight=args.min_abs_weight,
        )

        summary_rows_a.append(s_a)
        summary_rows_b.append(s_b)
        node_rows.extend(n_a)
        node_rows.extend(n_b)
        edge_rows.extend(e_a)
        edge_rows.extend(e_b)

        row = {
            "sample_id": sid,
            "bucket": bucket,
            "node_overlap_jaccard": _jaccard(set_n_a, set_n_b),
            "edge_overlap_jaccard": _jaccard(set_e_a, set_e_b),
        }
        numeric_keys = [
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
            "traced_total_path_mass",
            "traced_min_abs_weight",
            "traced_max_abs_weight",
        ]
        row["a_target_token_id"] = s_a["target_token_id"]
        row["b_target_token_id"] = s_b["target_token_id"]
        for k in numeric_keys:
            av = float(s_a[k])
            bv = float(s_b[k])
            row[f"a_{k}"] = av
            row[f"b_{k}"] = bv
            row[f"delta_{k}"] = av - bv
        sample_compare_rows.append(row)

        if i % max(1, args.log_every) == 0 or i == total:
            print(f"[progress] {i}/{total} ({i/total*100:.1f}%)")

    _write_csv(
        out_dir / "run_A_summary.csv",
        summary_rows_a,
        fieldnames=list(summary_rows_a[0].keys()),
    )
    _write_csv(
        out_dir / "run_B_summary.csv",
        summary_rows_b,
        fieldnames=list(summary_rows_b[0].keys()),
    )
    _write_csv(
        out_dir / "sample_compare_controlled.csv",
        sample_compare_rows,
        fieldnames=list(sample_compare_rows[0].keys()),
    )
    _write_csv(
        out_dir / "nodes_detailed_controlled.csv",
        node_rows,
        fieldnames=list(node_rows[0].keys()),
    )
    _write_csv(
        out_dir / "edges_detailed_controlled.csv",
        edge_rows,
        fieldnames=list(edge_rows[0].keys()),
    )

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in sample_compare_rows:
        by_bucket[r["bucket"]].append(r)

    summary_bucket_rows: list[dict] = []
    summary_metrics = [
        "node_overlap_jaccard",
        "edge_overlap_jaccard",
        "a_target_top3_concentration",
        "b_target_top3_concentration",
        "delta_target_top3_concentration",
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
    ]
    for bucket, rows in sorted(by_bucket.items(), key=lambda x: x[0]):
        out = {"bucket": bucket, "count": len(rows)}
        for m in summary_metrics:
            out[m] = _mean([float(x[m]) for x in rows])
        summary_bucket_rows.append(out)

    out_all = {"bucket": "__all__", "count": len(sample_compare_rows)}
    for m in summary_metrics:
        out_all[m] = _mean([float(x[m]) for x in sample_compare_rows])
    summary_bucket_rows.append(out_all)

    _write_csv(
        out_dir / "bucket_summary_controlled.csv",
        summary_bucket_rows,
        fieldnames=list(summary_bucket_rows[0].keys()),
    )

    print("[done] selected_samples:", len(selected_with_files))
    print("[done] dropped_missing_pt:", dropped)
    print("[done] sample_compare:", out_dir / "sample_compare_controlled.csv")
    print("[done] bucket_summary:", out_dir / "bucket_summary_controlled.csv")
    print("[done] nodes:", out_dir / "nodes_detailed_controlled.csv")
    print("[done] edges:", out_dir / "edges_detailed_controlled.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
