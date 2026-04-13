#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _node_score(node: dict, deg_score: float) -> float:
    influence = float(node.get("influence", 0.0) or 0.0)
    activation = float(node.get("activation", 0.0) or 0.0)
    return deg_score + 3.0 * influence + 0.05 * activation


def _bucket_x(node: dict, max_layer: int) -> int:
    ftype = node.get("feature_type", "")
    if ftype == "embedding":
        return 0
    if ftype == "logit":
        return max_layer + 2
    layer = node.get("layer", "0")
    try:
        return int(layer) + 1
    except Exception:  # noqa: BLE001
        return 1


def _norm(v: float, lo: float, hi: float, eps: float = 1e-8) -> float:
    return (v - lo) / max(hi - lo, eps)


def render_snapshot(
    input_json: Path,
    output_png: Path,
    max_nodes: int = 260,
    max_edges: int = 2400,
    label_top: int = 24,
) -> None:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    if not nodes or not links:
        raise RuntimeError("Graph JSON has no nodes or links.")

    node_by_id = {n["node_id"]: n for n in nodes}
    deg_score = defaultdict(float)
    for e in links:
        w = abs(float(e.get("weight", 0.0) or 0.0))
        deg_score[e["source"]] += w
        deg_score[e["target"]] += w

    scored_nodes = []
    for n in nodes:
        nid = n["node_id"]
        scored_nodes.append((nid, _node_score(n, deg_score[nid])))
    scored_nodes.sort(key=lambda x: x[1], reverse=True)

    logit_nodes = [n["node_id"] for n in nodes if n.get("feature_type") == "logit"]
    keep_nodes = set(logit_nodes)
    for nid, _ in scored_nodes:
        keep_nodes.add(nid)
        if len(keep_nodes) >= max_nodes:
            break

    scored_edges = []
    for e in links:
        s, t = e["source"], e["target"]
        if s in keep_nodes and t in keep_nodes:
            w = float(e.get("weight", 0.0) or 0.0)
            scored_edges.append((abs(w), s, t, w))
    scored_edges.sort(key=lambda x: x[0], reverse=True)
    kept_edges = scored_edges[:max_edges]

    used_nodes = set()
    for _, s, t, _ in kept_edges:
        used_nodes.add(s)
        used_nodes.add(t)
    used_nodes |= set(logit_nodes)

    kept_nodes = [node_by_id[nid] for nid in used_nodes if nid in node_by_id]
    max_layer = 0
    for n in kept_nodes:
        try:
            max_layer = max(max_layer, int(n.get("layer", 0)))
        except Exception:  # noqa: BLE001
            pass

    buckets = defaultdict(list)
    for n in kept_nodes:
        buckets[_bucket_x(n, max_layer)].append(n)
    for x in buckets:
        buckets[x].sort(
            key=lambda n: (
                n.get("feature_type", ""),
                -float(n.get("influence", 0.0) or 0.0),
                -float(n.get("activation", 0.0) or 0.0),
            )
        )

    pos = {}
    for x, arr in buckets.items():
        m = len(arr)
        if m == 1:
            ys = [0.0]
        else:
            ys = [1.0 - 2.0 * (i / (m - 1)) for i in range(m)]
        for n, y in zip(arr, ys):
            pos[n["node_id"]] = (float(x), float(y))

    ftype_colors = {
        "embedding": "#3B82F6",
        "cross layer transcoder": "#10B981",
        "mlp reconstruction error": "#F59E0B",
        "logit": "#EF4444",
    }

    fig, ax = plt.subplots(figsize=(15, 8), dpi=180)

    if kept_edges:
        abs_vals = [a for a, _, _, _ in kept_edges]
        lo, hi = min(abs_vals), max(abs_vals)
        segs = []
        cols = []
        widths = []
        for a, s, t, w in kept_edges:
            if s not in pos or t not in pos:
                continue
            x1, y1 = pos[s]
            x2, y2 = pos[t]
            segs.append([(x1, y1), (x2, y2)])
            alpha = 0.15 + 0.55 * _norm(a, lo, hi)
            if w >= 0:
                cols.append((0.12, 0.35, 0.95, alpha))
            else:
                cols.append((0.85, 0.20, 0.20, alpha))
            widths.append(0.2 + 1.4 * _norm(a, lo, hi))
        lc = LineCollection(segs, colors=cols, linewidths=widths, zorder=1)
        ax.add_collection(lc)

    for ftype, color in ftype_colors.items():
        arr = [n for n in kept_nodes if n.get("feature_type") == ftype and n["node_id"] in pos]
        if not arr:
            continue
        xs = [pos[n["node_id"]][0] for n in arr]
        ys = [pos[n["node_id"]][1] for n in arr]
        size = 14
        if ftype == "logit":
            size = 90
        elif ftype == "embedding":
            size = 10
        ax.scatter(xs, ys, s=size, c=color, label=ftype, zorder=2, edgecolors="none")

    node_rank = sorted(
        [(n["node_id"], _node_score(n, deg_score[n["node_id"]])) for n in kept_nodes if n["node_id"] in pos],
        key=lambda x: x[1],
        reverse=True,
    )
    for nid, _ in node_rank[:label_top]:
        n = node_by_id[nid]
        x, y = pos[nid]
        ftype = n.get("feature_type", "")
        if ftype == "logit":
            label = f"logit:{n.get('feature', '')}"
        elif ftype == "embedding":
            label = f"emb@{n.get('ctx_idx', '')}"
        else:
            label = f"L{n.get('layer','?')}:{n.get('ctx_idx','?')}"
        ax.text(x + 0.12, y, label, fontsize=6, alpha=0.85)

    ax.set_title(
        f"Circuit Snapshot | nodes={len(kept_nodes)} edges={len(kept_edges)} (from {input_json.name})",
        fontsize=11,
    )
    ax.set_xlim(-0.8, max_layer + 2.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Layer Axis (Embedding -> Features/Error -> Logit)")
    ax.set_ylabel("Ranked Node Placement")
    ax.grid(alpha=0.15, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a static PNG snapshot from graph JSON.")
    parser.add_argument("--input", required=True, help="Path to graph json file.")
    parser.add_argument("--output", required=True, help="Path to output png.")
    parser.add_argument("--max-nodes", type=int, default=260, help="Max nodes to keep for plotting.")
    parser.add_argument("--max-edges", type=int, default=2400, help="Max edges to keep for plotting.")
    parser.add_argument("--label-top", type=int, default=24, help="How many top nodes to label.")
    args = parser.parse_args()

    render_snapshot(
        input_json=Path(args.input),
        output_png=Path(args.output),
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        label_top=args.label_top,
    )
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

