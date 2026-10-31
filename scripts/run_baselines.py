"""Run dissertation baseline evaluations for the mental-health benchmark.

README-style usage:

    # Keyword/rule baseline on the repaired 1000-case JSON:
    python scripts/run_baselines.py \
      --input telegram_bot/src/experiments/results/pipeline_eval_20260529_012609_repaired_20260529_014744.json \
      --output-dir results/baselines \
      --run-keyword

    # Run both keyword and single-prompt LLM baselines:
    python scripts/run_baselines.py \
      --input telegram_bot/src/experiments/results/pipeline_eval_20260529_012609_repaired_20260529_014744.json \
      --output-dir results/baselines \
      --text-col input_text \
      --expected-risk-col expected_risk \
      --expected-alert-col alert_expected \
      --category-col category \
      --language-col language \
      --run-keyword \
      --run-llm

Inputs:
    CSV, JSON array, or JSONL. Column names are auto-detected where possible,
    and can be overridden with CLI options.

Outputs:
    baseline_keyword_predictions.csv
    baseline_llm_predictions.csv
    baseline_comparison_metrics.json
    baseline_category_metrics.csv
    baseline_confusion_matrices.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional convenience dependency
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines"
RISK_LEVELS = ["low", "low-medium", "medium", "medium-high", "high"]
ACCEPTED_RISK_MATCHES = {
    "low": {"low", "low-medium"},
    "low-medium": {"low", "low-medium", "medium"},
    "medium": {"low-medium", "medium", "medium-high"},
    "medium-high": {"medium", "medium-high", "high"},
    "high": {"medium-high", "high"},
}


KEYWORD_GROUPS: dict[str, list[str]] = {
    "crisis": [
        "kill myself", "end my life", "want to die", "i want to die",
        "i don't want to live", "do not want to live", "suicide", "suicidal",
        "self harm", "self-harm", "hurt myself", "harm myself", "take my life",
        "no reason to live",
        "не хочу жить", "хочу умереть", "покончить с собой", "убить себя",
        "суицид", "суицидальные мысли", "причинить себе вред", "навредить себе",
        "нет смысла жить",
        "өмір сүргім келмейді", "өлгім келеді", "өзімді өлтір",
        "өзіме қол жұмса", "өмірдің мәні жоқ",
    ],
    "depressive": [
        "hopeless", "empty", "worthless", "meaningless", "no meaning",
        "alone", "lonely", "can't get out of bed", "nothing matters",
        "tired of everything",
        "безнадежно", "безнадёжно", "пустота", "пусто", "нет сил",
        "нет смысла", "одиноко", "ничего не хочется", "устал от всего",
        "устала от всего", "апатия",
        "жалғыз", "ештеңе қаламаймын", "шаршадым", "мәні жоқ", "үміт жоқ",
    ],
    "anxiety": [
        "anxious", "anxiety", "panic", "panic attack", "worried", "worry",
        "fear", "scared", "can't breathe",
        "тревога", "тревожно", "паника", "паническая атака", "страх",
        "боюсь", "переживаю",
        "уайым", "қорқамын", "мазасыз", "үрей",
    ],
    "stress": [
        "stress", "stressed", "burnout", "burned out", "overwhelmed",
        "tired", "exhausted", "too much work",
        "стресс", "устал", "устала", "выгорел", "выгорела", "завал",
        "слишком много дел", "нет отдыха",
        "стресс", "шаршадым", "жұмыс көп", "демалыс жоқ",
    ],
}

KEYWORD_REGEX: dict[str, list[re.Pattern[str]]] = {
    "crisis": [
        re.compile(r"\b(i\s+)?want\s+to\s+die\b"),
        re.compile(r"\b(kill|hurt|harm)\s+myself\b"),
        re.compile(r"\b(no|zero)\s+reason\s+to\s+live\b"),
        re.compile(r"\bself[-\s]?harm\b"),
        re.compile(r"не\s+хоч\w*\s+жить"),
        re.compile(r"хоч\w*\s+умереть"),
        re.compile(r"поконч\w*\s+с\s+собой"),
        re.compile(r"өмір\s+сүргім\s+келмейді"),
    ],
    "depressive": [
        re.compile(r"can'?t\s+get\s+out\s+of\s+bed"),
        re.compile(r"nothing\s+matters"),
        re.compile(r"нет\s+сил"),
        re.compile(r"нет\s+смысла"),
        re.compile(r"ничего\s+не\s+хоч\w*"),
        re.compile(r"үміт\s+жоқ"),
    ],
    "anxiety": [
        re.compile(r"panic\s+attack"),
        re.compile(r"can'?t\s+breathe"),
        re.compile(r"паническ\w*\s+атак\w*"),
        re.compile(r"тревожн\w*"),
    ],
    "stress": [
        re.compile(r"too\s+much\s+work"),
        re.compile(r"burn(?:ed)?\s+out"),
        re.compile(r"слишком\s+много\s+дел"),
        re.compile(r"демалыс\s+жоқ"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run keyword and single-prompt LLM baselines.")
    parser.add_argument("--input", type=Path, default=None, help="Benchmark file: CSV, JSON, or JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-col", default="")
    parser.add_argument("--expected-risk-col", default="")
    parser.add_argument("--expected-alert-col", default="")
    parser.add_argument("--category-col", default="")
    parser.add_argument("--language-col", default="")
    parser.add_argument("--run-keyword", action="store_true")
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--model", default=os.getenv("MODEL", "gpt-6.1-sol"))
    parser.add_argument("--llm-cache", type=Path, default=None)
    parser.add_argument("--llm-limit", type=int, default=0, help="Optional cap for LLM calls. 0 means all.")
    return parser.parse_args()


def load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "telegram_bot" / ".env", override=True)


def find_default_input() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "telegram_bot" / "src" / "experiments" / "results").glob(
            "pipeline_eval_*repaired*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    candidates = sorted(
        PROJECT_ROOT.glob("**/*evaluation*scenarios*1000*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No default benchmark found. Pass --input explicitly.")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} is not an object.")
                rows.append(row)
        return rows
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file: {exc}") from exc
        if isinstance(data, list):
            if not all(isinstance(row, dict) for row in data):
                raise ValueError("JSON list must contain objects.")
            return data
        raise ValueError("JSON input must be an array of objects.")
    raise ValueError(f"Unsupported input format: {suffix}")


def pick_column(rows: list[dict[str, Any]], explicit: str, candidates: list[str], label: str) -> str:
    if not rows:
        raise ValueError("Input has no rows.")
    keys = set().union(*(row.keys() for row in rows[:50]))
    if explicit:
        if explicit not in keys:
            raise ValueError(f"{label} column '{explicit}' not found. Available columns: {sorted(keys)}")
        return explicit
    for name in candidates:
        if name in keys:
            return name
    raise ValueError(f"Could not auto-detect {label} column. Available columns: {sorted(keys)}")


def normalize_risk(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "lowmedium": "low-medium",
        "mediumhigh": "medium-high",
        "med": "medium",
    }
    return aliases.get(text, text)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "да", "истина"}


def keyword_predict(text: str) -> tuple[str, bool]:
    lowered = text.lower()
    matched = set()
    for group, keywords in KEYWORD_GROUPS.items():
        if any(keyword in lowered for keyword in keywords):
            matched.add(group)
        if any(pattern.search(lowered) for pattern in KEYWORD_REGEX.get(group, [])):
            matched.add(group)

    if "crisis" in matched:
        return "high", True
    if "depressive" in matched:
        return "medium-high", False
    if "anxiety" in matched:
        return "medium", False
    if "stress" in matched:
        return "low-medium", False
    return "low", False


def risk_matches(expected: str, predicted: str) -> bool:
    expected = normalize_risk(expected)
    predicted = normalize_risk(predicted)
    return predicted in ACCEPTED_RISK_MATCHES.get(expected, {expected})


def evaluate_prediction(row: dict[str, Any], expected_risk_col: str, expected_alert_col: str, predicted_risk: str, predicted_alert: bool) -> dict[str, Any]:
    expected_risk = normalize_risk(row.get(expected_risk_col))
    expected_alert = parse_bool(row.get(expected_alert_col))
    risk_match = risk_matches(expected_risk, predicted_risk)
    alert_match = predicted_alert == expected_alert
    return {
        "predicted_risk": normalize_risk(predicted_risk),
        "predicted_alert": bool(predicted_alert),
        "risk_match": risk_match,
        "alert_match": alert_match,
        "overall_pass": risk_match and alert_match,
    }


def load_llm_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = item.get("key")
            if key:
                cache[key] = item
    return cache


def append_llm_cache(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def cache_key(model: str, text: str) -> str:
    payload = json.dumps({"model": model, "text": text}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_llm_json(content: str) -> dict[str, Any]:
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    data = json.loads(raw)
    risk = normalize_risk(data.get("risk_level"))
    if risk not in RISK_LEVELS:
        raise ValueError(f"Invalid risk_level from LLM: {risk}")
    return {"risk_level": risk, "alert_required": parse_bool(data.get("alert_required"))}


def llm_predict(client: Any, model: str, text: str) -> tuple[str, bool]:
    system = (
        "You are a conservative conversational risk-signal classifier for prototype evaluation.\n"
        "You are not a clinician and must not diagnose mental disorders.\n"
        "Classify only the risk signal expressed in the given text.\n"
        "Return strict JSON only."
    )
    user = (
        "Classify the following user text into one risk level:\n"
        "- low\n"
        "- low-medium\n"
        "- medium\n"
        "- medium-high\n"
        "- high\n\n"
        "Also decide whether a specialist alert is required.\n\n"
        "Alert should be true mainly for explicit or strongly implied self-harm, suicidal intent, "
        "immediate danger, abuse-related emergency, or severe crisis escalation.\n"
        "Depressive, anxious, or stressed language without explicit danger may have elevated risk "
        "but should not automatically create an urgent alert.\n\n"
        "Return only:\n"
        "{\n"
        '  "risk_level": "...",\n'
        '  "alert_required": true/false\n'
        "}\n\n"
        f"Text:\n{text}"
    )
    # Reasoning models (gpt-5+, gpt-6+, o-series) reject `temperature`.
    if model.lower().startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        sampling = {"reasoning_effort": os.getenv("REASONING_EFFORT", "low")}
    else:
        sampling = {"temperature": 0}
    response = client.chat.completions.create(
        model=model,
        **sampling,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    parsed = parse_llm_json(response.choices[0].message.content or "")
    return parsed["risk_level"], parsed["alert_required"]


def init_openai_client() -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("The openai package is not available. Install project requirements first.") from exc
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env/telegram_bot/.env or export it.")
    return OpenAI()


def add_original_and_prediction(row: dict[str, Any], prediction: dict[str, Any], method: str, error: str = "") -> dict[str, Any]:
    output = dict(row)
    output.update(prediction)
    output["method"] = method
    if error:
        output["prediction_error"] = error
    return output


def run_keyword(rows: list[dict[str, Any]], cols: dict[str, str]) -> list[dict[str, Any]]:
    predictions = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        if index == 1 or index % 100 == 0 or index == total:
            print(f"[keyword] {index}/{total}", flush=True)
        risk, alert = keyword_predict(str(row.get(cols["text"], "")))
        evaluation = evaluate_prediction(row, cols["expected_risk"], cols["expected_alert"], risk, alert)
        predictions.append(add_original_and_prediction(row, evaluation, "keyword"))
    return predictions


def run_llm(rows: list[dict[str, Any]], cols: dict[str, str], output_dir: Path, model: str, cache_path: Path | None, limit: int = 0) -> list[dict[str, Any]]:
    cache_path = cache_path or output_dir / "baseline_llm_cache.jsonl"
    cache = load_llm_cache(cache_path)
    client = init_openai_client()
    predictions = []
    total = len(rows) if not limit else min(limit, len(rows))
    for index, row in enumerate(rows, start=1):
        if limit and index > limit:
            break
        text = str(row.get(cols["text"], ""))
        key = cache_key(model, text)
        print(f"[llm] {index}/{total} id={row.get('id', index)}", flush=True)
        error = ""
        if key in cache:
            risk = normalize_risk(cache[key]["risk_level"])
            alert = parse_bool(cache[key]["alert_required"])
        else:
            try:
                risk, alert = llm_predict(client, model, text)
                append_llm_cache(cache_path, {"key": key, "model": model, "text": text, "risk_level": risk, "alert_required": alert})
            except Exception as exc:
                risk, alert = "error", False
                error = f"{type(exc).__name__}: {exc}"
                print(f"    -> {error}", flush=True)
                time.sleep(1)
        evaluation = evaluate_prediction(row, cols["expected_risk"], cols["expected_alert"], risk, alert)
        predictions.append(add_original_and_prediction(row, evaluation, "llm", error=error))
    return predictions


def alert_counts(rows: list[dict[str, Any]], expected_alert_col: str) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        expected = parse_bool(row.get(expected_alert_col))
        predicted = parse_bool(row.get("predicted_alert"))
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif not expected and not predicted:
            tn += 1
        else:
            fn += 1
    return {
        "true_positive_alerts": tp,
        "false_positive_alerts": fp,
        "true_negative_alerts": tn,
        "false_negative_alerts": fn,
    }


def compute_metrics(rows: list[dict[str, Any]], expected_alert_col: str) -> dict[str, Any]:
    total = len(rows)
    counts = alert_counts(rows, expected_alert_col)
    tp = counts["true_positive_alerts"]
    fp = counts["false_positive_alerts"]
    tn = counts["true_negative_alerts"]
    fn = counts["false_negative_alerts"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "total_cases": total,
        "risk_match_rate": sum(parse_bool(row.get("risk_match")) for row in rows) / total if total else 0.0,
        "alert_accuracy": sum(parse_bool(row.get("alert_match")) for row in rows) / total if total else 0.0,
        "overall_pass_rate": sum(parse_bool(row.get("overall_pass")) for row in rows) / total if total else 0.0,
        "alert_precision": precision,
        "alert_recall": recall,
        "alert_f1": f1,
        **counts,
        "expected_alerts": tp + fn,
        "actual_alerts": tp + fp,
    }


def category_metrics(prediction_sets: dict[str, list[dict[str, Any]]], category_col: str, expected_alert_col: str) -> list[dict[str, Any]]:
    rows = []
    for method, predictions in prediction_sets.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped[str(row.get(category_col, "unknown") or "unknown")].append(row)
        for category, items in sorted(grouped.items()):
            metric = compute_metrics(items, expected_alert_col)
            metric["method"] = method
            metric["category"] = category
            rows.append(metric)
    return rows


def confusion_matrices(prediction_sets: dict[str, list[dict[str, Any]]], expected_risk_col: str) -> dict[str, Any]:
    matrices = {}
    for method, predictions in prediction_sets.items():
        matrix: dict[str, dict[str, int]] = {expected: {pred: 0 for pred in RISK_LEVELS + ["error"]} for expected in RISK_LEVELS}
        for row in predictions:
            expected = normalize_risk(row.get(expected_risk_col))
            predicted = normalize_risk(row.get("predicted_risk"))
            if expected not in matrix:
                matrix[expected] = {pred: 0 for pred in RISK_LEVELS + ["error"]}
            if predicted not in matrix[expected]:
                matrix[expected][predicted] = 0
            matrix[expected][predicted] += 1
        matrices[method] = matrix
    return matrices


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(overall: dict[str, dict[str, Any]]) -> None:
    print()
    print("method,total,risk_match,alert_acc,overall,precision,recall,f1")
    for method, metric in overall.items():
        print(
            f"{method},{metric['total_cases']},"
            f"{metric['risk_match_rate']:.3f},"
            f"{metric['alert_accuracy']:.3f},"
            f"{metric['overall_pass_rate']:.3f},"
            f"{metric['alert_precision']:.3f},"
            f"{metric['alert_recall']:.3f},"
            f"{metric['alert_f1']:.3f}"
        )


def main() -> int:
    args = parse_args()
    load_env()
    input_path = args.input or find_default_input()
    rows = load_rows(input_path)
    if not rows:
        print("No rows found in input.", file=sys.stderr)
        return 2

    if not args.run_keyword and not args.run_llm:
        args.run_keyword = True

    cols = {
        "text": pick_column(rows, args.text_col, ["input_text", "text", "message", "user_message"], "text"),
        "expected_risk": pick_column(rows, args.expected_risk_col, ["expected_risk", "expected_risk_level", "risk_label"], "expected risk"),
        "expected_alert": pick_column(rows, args.expected_alert_col, ["expected_alert", "alert_expected", "expected_alert_required"], "expected alert"),
        "category": pick_column(rows, args.category_col, ["category", "scenario_category", "label_category"], "category"),
        "language": pick_column(rows, args.language_col, ["language", "lang", "locale"], "language"),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input: {input_path}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {cols}")
    print(f"Output: {output_dir}")

    prediction_sets: dict[str, list[dict[str, Any]]] = {}
    if args.run_keyword:
        keyword_rows = run_keyword(rows, cols)
        prediction_sets["keyword"] = keyword_rows
        write_csv(output_dir / "baseline_keyword_predictions.csv", keyword_rows)

    if args.run_llm:
        try:
            llm_rows = run_llm(rows, cols, output_dir, args.model, args.llm_cache, limit=args.llm_limit)
            prediction_sets["llm"] = llm_rows
            write_csv(output_dir / "baseline_llm_predictions.csv", llm_rows)
        except Exception as exc:
            print(f"LLM baseline skipped/failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    overall = {
        method: compute_metrics(predictions, cols["expected_alert"])
        for method, predictions in prediction_sets.items()
    }
    cat_rows = category_metrics(prediction_sets, cols["category"], cols["expected_alert"])
    matrices = confusion_matrices(prediction_sets, cols["expected_risk"])

    (output_dir / "baseline_comparison_metrics.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "baseline_category_metrics.csv", cat_rows)
    (output_dir / "baseline_confusion_matrices.json").write_text(
        json.dumps(matrices, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary_table(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
