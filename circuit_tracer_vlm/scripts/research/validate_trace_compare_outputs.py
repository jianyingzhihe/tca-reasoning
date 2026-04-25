#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_FILES = [
    "selected_samples.csv",
    "run_A_summary.csv",
    "run_B_summary.csv",
    "sample_compare_controlled.csv",
    "bucket_summary_controlled.csv",
    "nodes_detailed_controlled.csv",
    "edges_detailed_controlled.csv",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _to_int(value: str, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"invalid int for {field}: {value!r}") from exc


def _to_float(value: str, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"invalid float for {field}: {value!r}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate outputs from trace_compare_ab_controlled.py."
    )
    parser.add_argument("--out-dir", required=True, help="Directory containing compare CSV outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    _require(out_dir.exists(), f"missing out dir: {out_dir}")

    paths = {name: out_dir / name for name in REQUIRED_FILES}
    for name, path in paths.items():
        _require(path.exists(), f"missing file: {path}")

    selected_rows = _read_csv(paths["selected_samples.csv"])
    run_a_rows = _read_csv(paths["run_A_summary.csv"])
    run_b_rows = _read_csv(paths["run_B_summary.csv"])
    sample_compare_rows = _read_csv(paths["sample_compare_controlled.csv"])
    bucket_summary_rows = _read_csv(paths["bucket_summary_controlled.csv"])
    node_rows = _read_csv(paths["nodes_detailed_controlled.csv"])
    edge_rows = _read_csv(paths["edges_detailed_controlled.csv"])

    _require(selected_rows, "selected_samples.csv is empty")
    _require(run_a_rows, "run_A_summary.csv is empty")
    _require(run_b_rows, "run_B_summary.csv is empty")
    _require(sample_compare_rows, "sample_compare_controlled.csv is empty")
    _require(bucket_summary_rows, "bucket_summary_controlled.csv is empty")
    _require(node_rows, "nodes_detailed_controlled.csv is empty")
    _require(edge_rows, "edges_detailed_controlled.csv is empty")

    selected_ids = {(r.get("sample_id") or "").strip() for r in selected_rows}
    selected_ids.discard("")
    _require(selected_ids, "selected_samples.csv has no sample_id values")

    sample_compare_ids = {(r.get("sample_id") or "").strip() for r in sample_compare_rows}
    _require(
        sample_compare_ids == selected_ids,
        f"sample ids mismatch: selected={len(selected_ids)} compare={len(sample_compare_ids)}",
    )

    def _index_summary(rows: list[dict[str, str]], run_name: str) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            sid = (row.get("sample_id") or "").strip()
            run = (row.get("run") or "").strip()
            _require(sid in selected_ids, f"{run_name} contains unexpected sample_id={sid}")
            _require(run == run_name, f"{run_name} row has wrong run field: {run!r}")
            _require(sid not in out, f"{run_name} duplicate sample_id={sid}")
            out[sid] = row
        _require(set(out) == selected_ids, f"{run_name} summary does not cover all selected samples")
        return out

    run_a = _index_summary(run_a_rows, "A")
    run_b = _index_summary(run_b_rows, "B")

    node_key_to_row: dict[tuple[str, str, str], dict[str, str]] = {}
    node_counts: dict[tuple[str, str], int] = defaultdict(int)
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in node_rows:
        sid = (row.get("sample_id") or "").strip()
        run = (row.get("run") or "").strip()
        node_id = (row.get("node_id") or "").strip()
        _require(sid in selected_ids, f"node row has unexpected sample_id={sid}")
        _require(run in {"A", "B"}, f"node row has invalid run={run!r}")
        _require(node_id, f"node row missing node_id for sample_id={sid} run={run}")
        key = (sid, run, node_id)
        _require(key not in node_key_to_row, f"duplicate node row for {key}")
        node_key_to_row[key] = row
        node_counts[(sid, run)] += 1

        _to_int(row["node_idx"], "node_idx")
        _to_int(row["layer"], "layer")
        if row.get("pos", "") != "":
            _to_int(row["pos"], "pos")
        _to_float(row["path_mass_best"], "path_mass_best")

    for row in edge_rows:
        sid = (row.get("sample_id") or "").strip()
        run = (row.get("run") or "").strip()
        src_node = (row.get("src_node") or "").strip()
        dst_node = (row.get("dst_node") or "").strip()
        _require(sid in selected_ids, f"edge row has unexpected sample_id={sid}")
        _require(run in {"A", "B"}, f"edge row has invalid run={run!r}")
        _require(src_node and dst_node, f"edge row missing src/dst node for sample_id={sid} run={run}")
        _require(
            (sid, run, src_node) in node_key_to_row,
            f"edge src_node missing from nodes file: sample_id={sid} run={run} src={src_node}",
        )
        _require(
            (sid, run, dst_node) in node_key_to_row,
            f"edge dst_node missing from nodes file: sample_id={sid} run={run} dst={dst_node}",
        )
        edge_counts[(sid, run)] += 1

        _to_int(row["depth"], "depth")
        _to_int(row["src_idx"], "src_idx")
        _to_int(row["dst_idx"], "dst_idx")
        _to_float(row["weight"], "weight")
        _to_float(row["abs_weight"], "abs_weight")
        _to_float(row["local_ratio"], "local_ratio")
        _to_float(row["dst_path_mass"], "dst_path_mass")
        _to_float(row["path_mass"], "path_mass")

    for sid in sorted(selected_ids):
        for run_name, summary_map in (("A", run_a), ("B", run_b)):
            summary = summary_map[sid]
            traced_nodes = _to_int(summary["traced_nodes"], "traced_nodes")
            traced_edges = _to_int(summary["traced_edges"], "traced_edges")
            _require(
                node_counts[(sid, run_name)] == traced_nodes,
                f"node count mismatch for sample_id={sid} run={run_name}: "
                f"summary={traced_nodes} actual={node_counts[(sid, run_name)]}",
            )
            _require(
                edge_counts[(sid, run_name)] == traced_edges,
                f"edge count mismatch for sample_id={sid} run={run_name}: "
                f"summary={traced_edges} actual={edge_counts[(sid, run_name)]}",
            )
            _to_int(summary["target_token_id"], "target_token_id")
            _to_int(summary["n_layers"], "n_layers")
            _to_int(summary["n_total_nodes"], "n_total_nodes")
            _to_int(summary["n_feature_nodes"], "n_feature_nodes")
            _to_int(summary["n_nonzero_edges"], "n_nonzero_edges")

    bucket_counts_from_compare: dict[str, int] = defaultdict(int)
    for row in sample_compare_rows:
        sid = (row.get("sample_id") or "").strip()
        bucket = (row.get("bucket") or "").strip()
        _require(sid in selected_ids, f"sample_compare has unexpected sample_id={sid}")
        _require(bucket, f"sample_compare missing bucket for sample_id={sid}")
        bucket_counts_from_compare[bucket] += 1
        _to_float(row["node_overlap_jaccard"], "node_overlap_jaccard")
        _to_float(row["edge_overlap_jaccard"], "edge_overlap_jaccard")

    bucket_summary = {(r.get("bucket") or "").strip(): r for r in bucket_summary_rows}
    _require("__all__" in bucket_summary, "bucket_summary_controlled.csv missing __all__ row")
    _require(
        _to_int(bucket_summary["__all__"]["count"], "bucket_summary.__all__.count")
        == len(sample_compare_rows),
        "bucket_summary __all__ count does not match sample_compare row count",
    )

    for bucket, count in bucket_counts_from_compare.items():
        _require(bucket in bucket_summary, f"bucket_summary missing bucket={bucket}")
        _require(
            _to_int(bucket_summary[bucket]["count"], f"bucket_summary[{bucket}].count") == count,
            f"bucket_summary count mismatch for bucket={bucket}",
        )

    print(f"[ok] out_dir={out_dir}")
    print(f"[ok] selected_samples={len(selected_ids)}")
    print(f"[ok] sample_compare_rows={len(sample_compare_rows)}")
    print(f"[ok] node_rows={len(node_rows)}")
    print(f"[ok] edge_rows={len(edge_rows)}")
    print(f"[ok] buckets={sorted(bucket_counts_from_compare)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
