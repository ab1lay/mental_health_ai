"""Repair an evaluation result JSON by rerunning only failed scenarios.

Usage:
    python telegram_bot/src/experiments/rerun_failed_pipeline_results.py \
        --input-json telegram_bot/src/experiments/results/pipeline_eval_20260529_012609.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from evaluate_pipeline_scenarios import (
    DEFAULT_OUTPUT_DIR,
    ScenarioResult,
    configure_imports_and_env,
    evaluate_one,
    print_summary,
)


def latest_result_json(results_dir: Path) -> Path | None:
    candidates = sorted(
        results_dir.glob("pipeline_eval_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    default_input = latest_result_json(DEFAULT_OUTPUT_DIR)
    parser = argparse.ArgumentParser(
        description="Rerun only errored rows from a previous pipeline evaluation JSON."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=default_input,
        help="Previous result JSON. Defaults to the latest pipeline_eval_*.json in results/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for repaired JSON/CSV. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--error-contains",
        default="",
        help="Rerun only error rows containing this substring. Empty means any non-empty error.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Attempts per failed row. Default: 2.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between attempts for the same row. Default: 2.0.",
    )
    parser.add_argument(
        "--fail-on-remaining-errors",
        action="store_true",
        help="Exit with code 1 if any rows still have errors after rerun.",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict]:
    if not path:
        raise FileNotFoundError("No input JSON was provided and no pipeline_eval_*.json was found.")
    if not path.exists():
        raise FileNotFoundError(f"Result JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Result JSON must contain a list of scenario result objects.")
    return data


def should_rerun(row: dict, error_contains: str) -> bool:
    error = str(row.get("error") or "").strip()
    if not error:
        return False
    if not error_contains:
        return True
    return error_contains.lower() in error.lower()


async def rerun_with_retries(row: dict, retries: int, retry_delay: float) -> ScenarioResult:
    attempts = max(1, retries)
    last_result: ScenarioResult | None = None
    for attempt in range(1, attempts + 1):
        last_result = await evaluate_one(row)
        if not last_result.error:
            return last_result
        if attempt < attempts:
            await asyncio.sleep(max(0.0, retry_delay))
    return last_result


async def repair_results(
    rows: list[dict],
    failed_indices: list[int],
    retries: int,
    retry_delay: float,
) -> list[dict]:
    repaired = list(rows)
    total = len(failed_indices)
    for number, row_index in enumerate(failed_indices, start=1):
        row = repaired[row_index]
        remaining = total - number
        print(
            f"[{number}/{total}] rerun id={row.get('id')} "
            f"category={row.get('category')} lang={row.get('language')} "
            f"remaining={remaining}",
            flush=True,
        )
        result = await rerun_with_retries(row, retries=retries, retry_delay=retry_delay)
        repaired[row_index] = asdict(result)
        status = "ok" if not result.error else f"error: {result.error}"
        print(f"    -> {status}", flush=True)
    return repaired


def write_repaired_results(rows: list[dict], output_dir: Path, source_json: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{source_json.stem}_repaired_{stamp}"
    json_path = output_dir / f"{base_name}.json"
    csv_path = output_dir / f"{base_name}.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def main() -> int:
    args = parse_args()
    configure_imports_and_env()

    rows = load_results(args.input_json)
    failed_indices = [
        index
        for index, row in enumerate(rows)
        if should_rerun(row, args.error_contains)
    ]

    print(f"Input JSON: {args.input_json}")
    print(f"Rows total: {len(rows)}")
    print(f"Rows selected for rerun: {len(failed_indices)}")
    if args.error_contains:
        print(f"Error filter: {args.error_contains}")

    if not failed_indices:
        print("Nothing to rerun.")
        return 0

    repaired = asyncio.run(
        repair_results(
            rows,
            failed_indices,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
    )
    json_path, csv_path = write_repaired_results(repaired, args.output_dir, args.input_json)

    remaining_errors = sum(1 for row in repaired if str(row.get("error") or "").strip())
    print()
    print_summary([ScenarioResult(**row) for row in repaired])
    print()
    print(f"Remaining errors: {remaining_errors}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")

    if args.fail_on_remaining_errors and remaining_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
