#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


SUCCESS_STATUSES = {"ok", "exists"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


@dataclass
class BucketResult:
    bucket: str
    run_tag: str
    status_ok: bool
    selected_count: int
    meta_a_count: int
    meta_b_count: int
    success_a_count: int
    success_b_count: int
    error_a_count: int
    error_b_count: int
    pt_a_count: int
    pt_b_count: int
    compare_present: bool
    compare_validated: bool
    queue_status: str
    queue_exit_code: str
    output_size_bytes: int
    messages: list[str]


def inspect_queue(
    *,
    run_tag_base: str,
    expected_buckets: list[str],
    outputs_root: Path,
    work_root: Path,
) -> tuple[list[BucketResult], Path]:
    queue_log_dir = outputs_root / f"{run_tag_base}_queue_logs"
    status_tsv = queue_log_dir / "bucket_status.tsv"
    queue_rows = _read_csv(status_tsv) if status_tsv.exists() else []
    queue_by_bucket = {(r.get("bucket") or "").strip(): r for r in queue_rows}

    results: list[BucketResult] = []
    for bucket in expected_buckets:
        run_tag = f"{run_tag_base}_{bucket}"
        out_dir = outputs_root / run_tag
        work_dir = work_root / run_tag
        selected_csv = work_dir / "selected_4bucket.csv"
        meta_a_csv = out_dir / "answer_aligned_meta_a.csv"
        meta_b_csv = out_dir / "answer_aligned_meta_b.csv"
        pt_a_dir = out_dir / "pt_a"
        pt_b_dir = out_dir / "pt_b"
        compare_dir = out_dir / run_tag

        messages: list[str] = []
        queue_row = queue_by_bucket.get(bucket, {})
        queue_status = (queue_row.get("status") or "").strip()
        queue_exit_code = (queue_row.get("exit_code") or "").strip()

        selected_rows = _read_csv(selected_csv) if selected_csv.exists() else []
        meta_a_rows = _read_csv(meta_a_csv) if meta_a_csv.exists() else []
        meta_b_rows = _read_csv(meta_b_csv) if meta_b_csv.exists() else []
        selected_ids = {(r.get("sample_id") or "").strip() for r in selected_rows}
        selected_ids.discard("")
        meta_a_ids = {(r.get("sample_id") or "").strip() for r in meta_a_rows}
        meta_a_ids.discard("")
        meta_b_ids = {(r.get("sample_id") or "").strip() for r in meta_b_rows}
        meta_b_ids.discard("")

        success_a_ids = {
            (r.get("sample_id") or "").strip()
            for r in meta_a_rows
            if (r.get("status") or "").strip() in SUCCESS_STATUSES
        }
        success_b_ids = {
            (r.get("sample_id") or "").strip()
            for r in meta_b_rows
            if (r.get("status") or "").strip() in SUCCESS_STATUSES
        }
        success_a_ids.discard("")
        success_b_ids.discard("")

        error_a_rows = [r for r in meta_a_rows if (r.get("status") or "").strip() not in SUCCESS_STATUSES]
        error_b_rows = [r for r in meta_b_rows if (r.get("status") or "").strip() not in SUCCESS_STATUSES]

        pt_a_ids = {p.stem for p in pt_a_dir.glob("*.pt")} if pt_a_dir.exists() else set()
        pt_b_ids = {p.stem for p in pt_b_dir.glob("*.pt")} if pt_b_dir.exists() else set()

        compare_present = compare_dir.exists()
        compare_validated = False
        if compare_present:
            required_compare_files = [
                compare_dir / "sample_compare_controlled.csv",
                compare_dir / "bucket_summary_controlled.csv",
                compare_dir / "nodes_detailed_controlled.csv",
                compare_dir / "edges_detailed_controlled.csv",
            ]
            compare_validated = all(p.exists() for p in required_compare_files)
            if not compare_validated:
                messages.append("compare dir exists but required compare csv files are missing")

        if not status_tsv.exists():
            messages.append(f"missing queue status file: {status_tsv}")
        elif bucket not in queue_by_bucket:
            messages.append(f"bucket not found in queue status: {bucket}")

        if not selected_csv.exists():
            messages.append(f"missing selected csv: {selected_csv}")
        if not meta_a_csv.exists():
            messages.append(f"missing meta A csv: {meta_a_csv}")
        if not meta_b_csv.exists():
            messages.append(f"missing meta B csv: {meta_b_csv}")

        if selected_ids and meta_a_ids != selected_ids:
            messages.append(
                f"meta A sample ids mismatch: selected={len(selected_ids)} meta_a={len(meta_a_ids)}"
            )
        if selected_ids and meta_b_ids != selected_ids:
            messages.append(
                f"meta B sample ids mismatch: selected={len(selected_ids)} meta_b={len(meta_b_ids)}"
            )

        if success_a_ids != pt_a_ids:
            messages.append(f"pt A count mismatch: success_meta={len(success_a_ids)} pt_files={len(pt_a_ids)}")
        if success_b_ids != pt_b_ids:
            messages.append(f"pt B count mismatch: success_meta={len(success_b_ids)} pt_files={len(pt_b_ids)}")

        if queue_status and queue_status != "ok":
            messages.append(f"queue status is not ok: {queue_status} exit_code={queue_exit_code}")
        if error_a_rows:
            messages.append(f"A has {len(error_a_rows)} non-success metadata rows")
        if error_b_rows:
            messages.append(f"B has {len(error_b_rows)} non-success metadata rows")

        status_ok = not messages
        results.append(
            BucketResult(
                bucket=bucket,
                run_tag=run_tag,
                status_ok=status_ok,
                selected_count=len(selected_rows),
                meta_a_count=len(meta_a_rows),
                meta_b_count=len(meta_b_rows),
                success_a_count=len(success_a_ids),
                success_b_count=len(success_b_ids),
                error_a_count=len(error_a_rows),
                error_b_count=len(error_b_rows),
                pt_a_count=len(pt_a_ids),
                pt_b_count=len(pt_b_ids),
                compare_present=compare_present,
                compare_validated=compare_validated,
                queue_status=queue_status,
                queue_exit_code=queue_exit_code,
                output_size_bytes=_dir_size_bytes(out_dir),
                messages=messages,
            )
        )

    return results, status_tsv


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate overnight answer-aligned bucket queue outputs.")
    parser.add_argument("--run-tag-base", required=True, help="Queue run tag base, without the trailing bucket name.")
    parser.add_argument("--buckets", default="A0_B1,A1_B1,A0_B0,A1_B0")
    parser.add_argument("--outputs-root", default="outputs/phase_ab/ab_answer_aligned")
    parser.add_argument("--work-root", default="research/work/ab_answer_aligned")
    args = parser.parse_args()

    expected_buckets = [x.strip() for x in args.buckets.split(",") if x.strip()]
    outputs_root = Path(args.outputs_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()

    results, status_tsv = inspect_queue(
        run_tag_base=args.run_tag_base,
        expected_buckets=expected_buckets,
        outputs_root=outputs_root,
        work_root=work_root,
    )

    print(f"[info] status_tsv={status_tsv}")
    ok_count = 0
    for res in results:
        line = (
            f"[bucket] {res.bucket} status={'ok' if res.status_ok else 'fail'} "
            f"queue={res.queue_status or '<missing>'}/{res.queue_exit_code or '?'} "
            f"selected={res.selected_count} "
            f"A={res.success_a_count}/{res.meta_a_count} pt={res.pt_a_count} "
            f"B={res.success_b_count}/{res.meta_b_count} pt={res.pt_b_count} "
            f"compare={'yes' if res.compare_validated else 'no'} "
            f"size={_human_bytes(res.output_size_bytes)}"
        )
        print(line)
        if res.status_ok:
            ok_count += 1
        else:
            for msg in res.messages:
                print(f"  - {msg}")

    if ok_count == len(results):
        print(f"[ok] all {ok_count}/{len(results)} buckets validated")
        return 0

    print(f"[fail] validated {ok_count}/{len(results)} buckets")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
