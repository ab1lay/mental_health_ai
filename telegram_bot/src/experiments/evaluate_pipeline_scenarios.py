"""Evaluate the mental-health pipeline on CSV scenarios.

Usage:
    python telegram_bot/src/experiments/evaluate_pipeline_scenarios.py \
        --input datasets/evaluation/mental_health_evaluation_scenarios_augmented_1000.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TELEGRAM_BOT_ROOT = PROJECT_ROOT / "telegram_bot"
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "evaluation" / "mental_health_evaluation_scenarios_augmented_1000.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "telegram_bot" / "src" / "experiments" / "results"


@dataclass
class ScenarioResult:
    id: str
    category: str
    language: str
    input_text: str
    expected_risk: str
    actual_risk: str
    risk_pass: bool
    alert_expected: bool
    alert_actual: bool
    alert_pass: bool
    overall_pass: bool
    emotion: str
    depression_score: float
    anxiety_score: float
    stress_score: float
    expected_behavior: str
    evaluation_focus: str
    response: str
    summary: str
    profile_note: str
    key_characteristics: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CSV mental-health evaluation scenarios through the project LangGraph pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Scenario CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV/JSON results. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of scenarios to run. 0 means all.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with code 1 if any scenario fails risk or alert checks.",
    )
    return parser.parse_args()


def configure_imports_and_env() -> None:
    """Load env files before importing modules that instantiate Settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(TELEGRAM_BOT_ROOT / ".env", override=True)

    # The graph does not need Telegram, but config.Settings requires it at import time.
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy-token-for-evaluation")

    if str(TELEGRAM_BOT_ROOT) not in sys.path:
        sys.path.insert(0, str(TELEGRAM_BOT_ROOT))


def read_scenarios(path: Path, limit: int = 0) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "id",
            "category",
            "language",
            "input_text",
            "expected_risk",
            "alert_expected",
            "expected_behavior",
            "evaluation_focus",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Scenario CSV is missing columns: {', '.join(missing)}")
        rows = list(reader)
    return rows[:limit] if limit and limit > 0 else rows


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "да"}


def acceptable_risks(expected: str) -> set[str]:
    normalized = str(expected).strip().lower().replace("-", "_")
    mapping = {
        "low": {"low"},
        "medium": {"medium"},
        "high": {"high"},
        "low_medium": {"low", "medium"},
        "medium_high": {"medium", "high"},
        "any": {"low", "medium", "high"},
    }
    return mapping.get(normalized, {normalized})


async def evaluate_one(row: dict[str, str]) -> ScenarioResult:
    from src.core.graph import compiled_graph, should_create_alert

    expected_risk = row["expected_risk"].strip().lower()
    alert_expected = parse_bool(row["alert_expected"])

    try:
        result: dict[str, Any] = await compiled_graph.ainvoke(
            {
                "text": row["input_text"],
                "history": [],
                "language": row["language"].strip().lower() or "en",
                "has_specialist": False,
                "preferences": {},
            }
        )
        analysis = result["analysis"]
        actual_risk = str(analysis.risk_level).strip().lower()
        alert_actual = should_create_alert(analysis)
        risk_pass = actual_risk in acceptable_risks(expected_risk)
        alert_pass = alert_actual == alert_expected

        return ScenarioResult(
            id=row["id"],
            category=row["category"],
            language=row["language"],
            input_text=row["input_text"],
            expected_risk=expected_risk,
            actual_risk=actual_risk,
            risk_pass=risk_pass,
            alert_expected=alert_expected,
            alert_actual=alert_actual,
            alert_pass=alert_pass,
            overall_pass=risk_pass and alert_pass,
            emotion=analysis.emotion,
            depression_score=float(analysis.depression_score),
            anxiety_score=float(analysis.anxiety_score),
            stress_score=float(analysis.stress_score),
            expected_behavior=row["expected_behavior"],
            evaluation_focus=row["evaluation_focus"],
            response=result.get("response", ""),
            summary=analysis.summary,
            profile_note=analysis.profile_note,
            key_characteristics=json.dumps(analysis.key_characteristics, ensure_ascii=False),
        )
    except Exception as exc:
        return ScenarioResult(
            id=row.get("id", ""),
            category=row.get("category", ""),
            language=row.get("language", ""),
            input_text=row.get("input_text", ""),
            expected_risk=expected_risk,
            actual_risk="error",
            risk_pass=False,
            alert_expected=alert_expected,
            alert_actual=False,
            alert_pass=not alert_expected,
            overall_pass=False,
            emotion="",
            depression_score=0.0,
            anxiety_score=0.0,
            stress_score=0.0,
            expected_behavior=row.get("expected_behavior", ""),
            evaluation_focus=row.get("evaluation_focus", ""),
            response="",
            summary="",
            profile_note="",
            key_characteristics="[]",
            error=f"{type(exc).__name__}: {exc}",
        )


async def evaluate_all(rows: list[dict[str, str]]) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}/{len(rows)}] scenario {row.get('id')} "
            f"({row.get('category')}, {row.get('language')})",
            flush=True,
        )
        results.append(await evaluate_one(row))
    return results


def write_results(results: list[ScenarioResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"pipeline_eval_{stamp}.csv"
    json_path = output_dir / f"pipeline_eval_{stamp}.json"

    rows = [asdict(item) for item in results]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def print_summary(results: list[ScenarioResult]) -> None:
    total = len(results)
    passed = sum(1 for item in results if item.overall_pass)
    risk_passed = sum(1 for item in results if item.risk_pass)
    alert_passed = sum(1 for item in results if item.alert_pass)
    failed = [item for item in results if not item.overall_pass]

    print()
    print(f"Total: {total}")
    print(f"Overall passed: {passed}/{total}")
    print(f"Risk passed: {risk_passed}/{total}")
    print(f"Alert passed: {alert_passed}/{total}")

    if failed:
        print()
        print("Mismatches:")
        for item in failed:
            print(
                f"- id={item.id} category={item.category} lang={item.language}: "
                f"expected_risk={item.expected_risk}, actual_risk={item.actual_risk}, "
                f"alert_expected={item.alert_expected}, alert_actual={item.alert_actual}"
            )
            if item.error:
                print(f"  error={item.error}")


def main() -> int:
    args = parse_args()
    configure_imports_and_env()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is required. Put it in .env or telegram_bot/.env, "
            "or export it before running this script.",
            file=sys.stderr,
        )
        return 2

    rows = read_scenarios(args.input, args.limit)
    if not rows:
        print("No scenarios found.", file=sys.stderr)
        return 2

    results = asyncio.run(evaluate_all(rows))
    csv_path, json_path = write_results(results, args.output_dir)
    print_summary(results)
    print()
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

    if args.fail_on_mismatch and any(not item.overall_pass for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
