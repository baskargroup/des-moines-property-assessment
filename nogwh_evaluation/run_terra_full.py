"""Run GPT-5.6 Terra on all 1,280 NOGWH images with the production P6 prompt.

Features:
  - Responses API with an explicit cache breakpoint after the stable P6 prefix
  - one sequential cache-writing request, then concurrent cache reads
  - append-only, resume-safe JSONL output
  - retry/backoff for 429, 5xx, timeouts, and network failures
  - $25 hard ceiling based on provider-reported usage
  - final metrics against original city scores and reviewer-corrected scores

Local paths come from the repository .env (see .env.example):
  NOGWH_IMAGES_DIR  folder with the 1,280 benchmark images (private City data)
  DSM_ENV_FILE      file holding the OpenAI key used for this campaign
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

from prompt import get_prompt_and_examples


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = HERE / "nogwh_manifest.csv"
load_dotenv(ROOT / ".env")
IMAGES_DIR = Path(os.getenv("NOGWH_IMAGES_DIR", ROOT / "data" / "private" / "nogwh_images"))
CORRECTED_CSV = HERE / "nogwh_updated_ground_truth_predictions.csv"
OUT = HERE / "predictions_full_terra.jsonl"
SUMMARY = HERE / "terra_full_summary.json"
CAMPAIGN_ENV = Path(os.getenv("DSM_ENV_FILE", ROOT / ".env"))
RUN_TAG = "base"

MODEL = "gpt-5.6-terra"
WORKERS = 12
MAX_ATTEMPTS = 5
HARD_CEILING_USD = 25.0
TAIL = "\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."

# Per 1M tokens, September 20, 2026.
PRICE = {"input": 2.0, "cached": 0.2, "cache_write": 2.5, "output": 12.0}

load_dotenv(CAMPAIGN_ENV, override=True)
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit(f"OPENAI_API_KEY missing in {CAMPAIGN_ENV}")


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


prompt_text, examples_raw = get_prompt_and_examples()
EXAMPLES = []
for path, score, description in examples_raw:
    p = Path(path).resolve()
    EXAMPLES.append((p, score, description, encode(p)))


def stable_prefix() -> list[dict]:
    parts: list[dict] = [{"type": "input_text", "text": prompt_text}]
    for path, score, description, encoded in EXAMPLES:
        parts.append(
            {
                "type": "input_text",
                "text": f"\n--- EXAMPLE (Score {score}) ---\n{description}",
            }
        )
        parts.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime(path)};base64,{encoded}",
                "detail": "auto",
            }
        )
    # This text was already in the original request. The metadata only marks
    # the end of reusable content; it does not change model-visible text.
    parts.append(
        {
            "type": "input_text",
            "text": TAIL,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    )
    return parts


PREFIX = stable_prefix()
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def request_body(image_path: Path) -> dict:
    parts = list(PREFIX)
    parts.append(
        {
            "type": "input_image",
            "image_url": f"data:{mime(image_path)};base64,{encode(image_path)}",
            "detail": "auto",
        }
    )
    return {
        "model": MODEL,
        "input": [{"role": "user", "content": parts}],
        "max_output_tokens": 8000,
        "reasoning": {"effort": "low"},
        "prompt_cache_key": f"nogwh-p6-terra-{RUN_TAG}",
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
    }


def output_text(data: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def parse(raw: str | None) -> dict | None:
    if not raw:
        return None
    s = raw.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0].strip()
    if not s.startswith("{"):
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j > i:
            s = s[i : j + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def usage_from(data: dict) -> dict:
    usage = data.get("usage", {})
    inp = usage.get("input_tokens_details", {})
    out = usage.get("output_tokens_details", {})
    reasoning = out.get("reasoning_tokens", 0) or 0
    return {
        "prompt_tokens": usage.get("input_tokens", 0) or 0,
        "cached_tokens": inp.get("cached_tokens", 0) or 0,
        "cache_write_tokens": inp.get("cache_write_tokens", 0) or 0,
        "output_tokens": max((usage.get("output_tokens", 0) or 0) - reasoning, 0),
        "reasoning_tokens": reasoning,
    }


def cost_usd(usage: dict) -> float:
    cached = usage["cached_tokens"]
    writes = usage["cache_write_tokens"]
    ordinary = max(usage["prompt_tokens"] - cached - writes, 0)
    output = usage["output_tokens"] + usage["reasoning_tokens"]
    return (
        ordinary * PRICE["input"]
        + cached * PRICE["cached"]
        + writes * PRICE["cache_write"]
        + output * PRICE["output"]
    ) / 1_000_000


def call(image_path: Path) -> tuple[str | None, dict, float, str | None]:
    body = request_body(image_path)
    error = None
    for attempt in range(MAX_ATTEMPTS):
        started = time.monotonic()
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=HEADERS,
                json=body,
                timeout=180,
            )
            elapsed = time.monotonic() - started
            if response.status_code == 200:
                data = response.json()
                raw = output_text(data)
                if raw:
                    return raw, usage_from(data), elapsed, None
                error = "empty response output"
            else:
                error = f"http_{response.status_code}: {response.text[:300]}"
                if (
                    response.status_code not in (408, 409, 429)
                    and response.status_code < 500
                ):
                    return None, {}, elapsed, error
        except requests.RequestException as exc:
            elapsed = time.monotonic() - started
            error = f"{type(exc).__name__}: {str(exc)[:250]}"
        if attempt < MAX_ATTEMPTS - 1:
            delay = min(60, 2 * (2**attempt)) * (0.5 + random.random())
            time.sleep(delay)
    return None, {}, elapsed, error


write_lock = threading.Lock()
budget_lock = threading.Lock()
spent = 0.0


def append(record: dict) -> None:
    with write_lock:
        with OUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def successful_ids() -> set[str]:
    done: set[str] = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("ok"):
                    done.add(record["image_id"])
    return done


def process(row: dict) -> dict:
    global spent
    image_path = IMAGES_DIR / row["filename"]
    raw, usage, elapsed, error = call(image_path)
    parsed = parse(raw)
    ok = parsed is not None and isinstance(parsed.get("score"), int)
    request_cost = cost_usd(usage) if usage else 0.0
    with budget_lock:
        spent += request_cost
        total_spent = spent
    record = {
        "image_id": row["image_id"],
        "filename": row["filename"],
        "image_path": str(image_path),
        "property_globalid": row["property_globalid"],
        "gt_score": int(row["gt_score"]),
        "gt_category": row["gt_category"],
        "model": MODEL,
        "ok": ok,
        "predicted_score": parsed.get("score") if parsed else None,
        "predicted_category": parsed.get("category") if parsed else None,
        "parsed": parsed,
        "raw_response": raw,
        "usage": usage,
        "cost_usd": round(request_cost, 6),
        "elapsed_sec": round(elapsed, 2),
        "error": error if error else (None if ok else "unparseable JSON"),
        "total_spent_after_usd": round(total_spent, 6),
    }
    append(record)
    return record


def c3(score: int | None) -> str | None:
    if score in (1, 2):
        return "H"
    if score == 3:
        return "I"
    if score in (4, 5):
        return "U"
    return None


def metrics(records: list[dict], truth: dict[str, int]) -> dict:
    pairs = [
        (record.get("predicted_score"), truth.get(record["filename"]))
        for record in records
    ]
    pairs = [
        (pred, gt)
        for pred, gt in pairs
        if pred in (1, 2, 3, 4, 5) and gt in (1, 2, 3, 4, 5)
    ]
    n = len(pairs)
    return {
        "n": n,
        "accuracy_5class_pct": round(
            100 * sum(pred == gt for pred, gt in pairs) / max(n, 1), 1
        ),
        "accuracy_3class_pct": round(
            100 * sum(c3(pred) == c3(gt) for pred, gt in pairs) / max(n, 1), 1
        ),
        "within_1_pct": round(
            100 * sum(abs(pred - gt) <= 1 for pred, gt in pairs) / max(n, 1), 1
        ),
        "mae": round(
            sum(abs(pred - gt) for pred, gt in pairs) / max(n, 1), 3
        ),
    }


def summarize(wall_seconds: float) -> dict:
    records = []
    with OUT.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("ok"):
                records.append(record)

    corrected_rows = list(
        csv.DictReader(CORRECTED_CSV.open(encoding="utf-8-sig"))
    )
    original_truth = {
        row["filename"]: int(row["original_ground_truth"])
        for row in corrected_rows
        if row["original_ground_truth"] in {"1", "2", "3", "4", "5"}
    }
    reviewed_rows = [
        row for row in corrected_rows if row["ground_truth_source"] == "returned_score"
    ]
    corrected_truth = {
        row["filename"]: int(row["updated_ground_truth"]) for row in reviewed_rows
    }
    reviewed_city_truth = {
        row["filename"]: int(row["original_ground_truth"]) for row in reviewed_rows
    }

    def mean_usage(key: str) -> int:
        return round(
            sum(record["usage"].get(key, 0) for record in records)
            / max(len(records), 1)
        )

    summary = {
        "model": MODEL,
        "records_ok": len(records),
        "records_failed": 1280 - len(records),
        "wall_seconds_this_run": round(wall_seconds, 1),
        "total_cost_usd": round(sum(r["cost_usd"] for r in records), 4),
        "cost_per_1000_usd": round(
            1000 * sum(r["cost_usd"] for r in records) / max(len(records), 1), 2
        ),
        "mean_latency_sec": round(
            sum(r["elapsed_sec"] for r in records) / max(len(records), 1), 2
        ),
        "mean_usage": {
            key: mean_usage(key)
            for key in (
                "prompt_tokens",
                "cached_tokens",
                "cache_write_tokens",
                "output_tokens",
                "reasoning_tokens",
            )
        },
        "cache_hit_rate": round(
            sum(r["usage"].get("cached_tokens", 0) for r in records)
            / max(sum(r["usage"].get("prompt_tokens", 0) for r in records), 1),
            4,
        ),
        "vs_original_city_all": metrics(records, original_truth),
        "vs_original_city_reviewed_subset": metrics(records, reviewed_city_truth),
        "vs_reviewer_corrected_subset": metrics(records, corrected_truth),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    global spent, OUT, SUMMARY, RUN_TAG
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="base",
        help="Independent run label; non-base labels write to separate JSONL and summary files.",
    )
    args = parser.parse_args()
    RUN_TAG = args.run_id
    if RUN_TAG != "base":
        safe_tag = "".join(c for c in RUN_TAG if c.isalnum() or c in ("-", "_"))
        if safe_tag != RUN_TAG or not safe_tag:
            raise SystemExit("--run-id may contain only letters, numbers, hyphens, and underscores")
        OUT = HERE / f"predictions_full_terra_{safe_tag}.jsonl"
        SUMMARY = HERE / f"terra_full_summary_{safe_tag}.json"

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    missing = [row["filename"] for row in rows if not (IMAGES_DIR / row["filename"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing; first: {missing[0]}")

    done = successful_ids()
    pending = [row for row in rows if row["image_id"] not in done]
    if OUT.exists():
        with OUT.open(encoding="utf-8") as handle:
            spent = sum(
                json.loads(line).get("cost_usd", 0)
                for line in handle
                if line.strip()
            )

    print(
        f"Terra NOGWH: total={len(rows)} done={len(done)} pending={len(pending)} "
        f"prior_spend=${spent:.2f} ceiling=${HARD_CEILING_USD:.2f}",
        flush=True,
    )
    if not pending:
        print(json.dumps(summarize(0), indent=2))
        return

    started = time.monotonic()
    # Exactly one cache-writing request before parallel cache reads.
    first = process(pending.pop(0))
    print(
        f"cache warm-up: ok={first['ok']} write={first['usage'].get('cache_write_tokens', 0)} "
        f"cached={first['usage'].get('cached_tokens', 0)} spent=${spent:.2f}",
        flush=True,
    )
    if not first["ok"]:
        raise SystemExit(f"cache warm-up request failed: {first['error']}")

    completed = 1
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process, row): row for row in pending}
        for future in as_completed(futures):
            record = future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(pending) + 1:
                elapsed = time.monotonic() - started
                rate = completed / max(elapsed, 1e-9)
                eta = (len(pending) + 1 - completed) / max(rate, 1e-9)
                print(
                    f"[{completed:>4}/{len(pending)+1}] ok={record['ok']} "
                    f"spent=${spent:.2f} rate={rate:.2f}/s eta={eta/60:.1f}m",
                    flush=True,
                )
            with budget_lock:
                if spent >= HARD_CEILING_USD:
                    for queued in futures:
                        queued.cancel()
                    raise SystemExit(
                        f"Hard ceiling reached: ${spent:.2f} >= ${HARD_CEILING_USD:.2f}"
                    )

    summary = summarize(time.monotonic() - started)
    print("\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
