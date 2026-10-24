"""Staged, resume-safe GPT-5.6 Terra citywide scoring campaign.

Stages are cumulative and manual:
  python run_terra_campaign.py --dry-run --limit 1000
  python run_terra_campaign.py --preflight --limit 1000  # uploads 5 examples; no inference
  python run_terra_campaign.py --limit 1000              # paid Stage 1
  python run_terra_campaign.py --limit 10000             # later, after approval

The five repeated calibration images are uploaded once to OpenAI Files and
referenced by file_id. Only the changing target photo is sent as base64.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=True)
API_KEY = os.getenv("OPENAI_API_KEY")
RESPONSES_URL = "https://api.openai.com/v1/responses"
INPUT_TOKENS_URL = "https://api.openai.com/v1/responses/input_tokens"
FILES_URL = "https://api.openai.com/v1/files"
TAIL_TEXT = "\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mime_for(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def response_text(data: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def usage_from(data: dict) -> dict:
    usage = data.get("usage", {})
    input_details = usage.get("input_tokens_details", {})
    output_details = usage.get("output_tokens_details", {})
    reasoning = output_details.get("reasoning_tokens", 0) or 0
    return {
        "prompt_tokens": usage.get("input_tokens", 0) or 0,
        "cached_tokens": input_details.get("cached_tokens", 0) or 0,
        "cache_write_tokens": input_details.get("cache_write_tokens", 0) or 0,
        "output_tokens": max((usage.get("output_tokens", 0) or 0) - reasoning, 0),
        "reasoning_tokens": reasoning,
    }


def cost_usd(usage: dict, prices: dict) -> float:
    cached = usage.get("cached_tokens", 0)
    writes = usage.get("cache_write_tokens", 0)
    ordinary = max(usage.get("prompt_tokens", 0) - cached - writes, 0)
    output = usage.get("output_tokens", 0) + usage.get("reasoning_tokens", 0)
    return (
        ordinary * prices["input"]
        + cached * prices["cached"]
        + writes * prices["cache_write"]
        + output * prices["output"]
    ) / 1_000_000


def class3(score: int | None) -> str | None:
    if score in (1, 2):
        return "Healthy"
    if score == 3:
        return "Intermediate"
    if score in (4, 5):
        return "Unhealthy"
    return None


class Budget:
    def __init__(self, ceiling: float, reserve_per_request: float, prior_spend: float):
        self.ceiling = ceiling
        self.reserve_per_request = reserve_per_request
        self.spent = prior_spend
        self.reserved = 0.0
        self.lock = threading.Lock()

    def reserve(self) -> bool:
        with self.lock:
            if self.spent + self.reserved + self.reserve_per_request > self.ceiling:
                return False
            self.reserved += self.reserve_per_request
            return True

    def commit(self, actual: float) -> None:
        with self.lock:
            self.reserved = max(0.0, self.reserved - self.reserve_per_request)
            self.spent += actual

    def release(self) -> None:
        with self.lock:
            self.reserved = max(0.0, self.reserved - self.reserve_per_request)


class Campaign:
    def __init__(self, config_path: Path, limit: int):
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.limit = limit
        self.run_id = self.cfg["run_id"]
        self.run_dir = HERE / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.run_dir / "predictions.jsonl"
        self.file_state_path = self.run_dir / "openai_example_files.json"
        self.manifest_path = HERE / self.cfg["manifest"]
        self.rows = self.load_manifest()[:limit]
        self.prefix_text, self.examples = self.load_prompt()
        self.write_lock = threading.Lock()
        self.abort = threading.Event()
        self.fatal_reason: str | None = None
        self.warning_emitted: set[float] = set()
        prior_spend = self.load_prior_spend()
        budget_cfg = self.cfg["budget"]
        self.budget = Budget(
            budget_cfg["hard_ceiling_usd"],
            budget_cfg["request_reserve_usd"],
            prior_spend,
        )

    def load_manifest(self) -> list[dict]:
        with self.manifest_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise SystemExit(f"Manifest is empty: {self.manifest_path}")
        return rows

    def load_prompt(self):
        import sys

        module_dir = (HERE / self.cfg["prompt_module_dir"]).resolve()
        sys.path.insert(0, str(module_dir))
        cwd = os.getcwd()
        os.chdir(module_dir)
        try:
            from prompt import get_prompt_and_examples

            text, raw_examples = get_prompt_and_examples()
            examples = [
                {
                    "path": str(Path(path).resolve()),
                    "score": score,
                    "description": description,
                }
                for path, score, description in raw_examples
            ]
        finally:
            os.chdir(cwd)
        return text, examples

    def all_records(self) -> list[dict]:
        if not self.predictions_path.exists():
            return []
        records = []
        with self.predictions_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def successful_ids(self) -> set[str]:
        return {
            record["image_id"]
            for record in self.all_records()
            if record.get("ok")
        }

    def load_prior_spend(self) -> float:
        return sum(record.get("cost_usd", 0) for record in self.all_records())

    def append(self, record: dict) -> None:
        with self.write_lock:
            with self.predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def ensure_example_files(self) -> dict[str, str]:
        state = {}
        if self.file_state_path.exists():
            state = json.loads(self.file_state_path.read_text(encoding="utf-8"))
        updated = {}
        auth = {"Authorization": f"Bearer {API_KEY}"}
        for example in self.examples:
            path = Path(example["path"])
            fingerprint = sha256(path)
            cached = state.get(path.name, {})
            file_id = cached.get("file_id") if cached.get("sha256") == fingerprint else None
            if file_id:
                check = requests.get(f"{FILES_URL}/{file_id}", headers=auth, timeout=60)
                if check.status_code != 200:
                    file_id = None
            if not file_id:
                with path.open("rb") as image:
                    response = requests.post(
                        FILES_URL,
                        headers=auth,
                        data={"purpose": "vision"},
                        files={"file": (path.name, image, mime_for(path))},
                        timeout=180,
                    )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to upload {path.name}: HTTP {response.status_code} "
                        f"{response.text[:300]}"
                    )
                file_id = response.json()["id"]
                print(f"uploaded example {path.name} -> {file_id}", flush=True)
            updated[path.name] = {"file_id": file_id, "sha256": fingerprint}
        self.file_state_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        return {name: value["file_id"] for name, value in updated.items()}

    def stable_prefix(self, file_ids: dict[str, str]) -> list[dict]:
        parts: list[dict] = [{"type": "input_text", "text": self.prefix_text}]
        for example in self.examples:
            path = Path(example["path"])
            parts.append(
                {
                    "type": "input_text",
                    "text": (
                        f"\n--- EXAMPLE (Score {example['score']}) ---\n"
                        f"{example['description']}"
                    ),
                }
            )
            parts.append(
                {
                    "type": "input_image",
                    "file_id": file_ids[path.name],
                    "detail": "auto",
                }
            )
        parts.append(
            {
                "type": "input_text",
                "text": TAIL_TEXT,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        )
        return parts

    def input_items(self, row: dict, prefix: list[dict]) -> list[dict]:
        image_path = Path(row["image_path"])
        target = {
            "type": "input_image",
            "image_url": f"data:{mime_for(image_path)};base64,{encode(image_path)}",
            "detail": "auto",
        }
        return [{"role": "user", "content": list(prefix) + [target]}]

    def request_body(self, row: dict, prefix: list[dict]) -> dict:
        return {
            "model": self.cfg["model"],
            "input": self.input_items(row, prefix),
            "max_output_tokens": self.cfg["max_output_tokens"],
            "reasoning": {"effort": self.cfg["reasoning_effort"]},
            "prompt_cache_key": self.cfg["prompt_cache_key"],
            "prompt_cache_options": {
                "mode": "explicit",
                "ttl": self.cfg["prompt_cache_ttl"],
            },
        }

    @staticmethod
    def output_text(data: dict) -> str:
        return "".join(
            part.get("text", "")
            for item in data.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )

    def call(self, row: dict, prefix: list[dict]) -> dict:
        prices = self.cfg["prices_per_million"]
        total_cost = 0.0
        attempt_usages = []
        last_error = None
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, self.cfg["max_attempts"] + 1):
            started = time.monotonic()
            try:
                response = requests.post(
                    RESPONSES_URL,
                    headers=headers,
                    json=self.request_body(row, prefix),
                    timeout=self.cfg["request_timeout_seconds"],
                )
                elapsed = time.monotonic() - started
            except requests.RequestException as exc:
                response = None
                elapsed = time.monotonic() - started
                last_error = f"{type(exc).__name__}: {str(exc)[:250]}"

            if response is not None and response.status_code == 200:
                data = response.json()
                usage = usage_from(data)
                attempt_cost = cost_usd(usage, prices)
                total_cost += attempt_cost
                attempt_usages.append(usage)
                raw = self.output_text(data)
                parsed = parse_json(raw)
                if parsed is not None and isinstance(parsed.get("score"), int):
                    return {
                        "ok": True,
                        "attempt": attempt,
                        "parsed": parsed,
                        "raw_response": raw,
                        "usage": usage,
                        "attempt_usages": attempt_usages,
                        "cost_usd": round(total_cost, 6),
                        "elapsed_sec": round(elapsed, 2),
                        "error": None,
                    }
                last_error = "unparseable JSON or missing integer score"
            elif response is not None:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                body = response.text.lower()
                billing_exhausted = response.status_code == 429 and (
                    "no credits" in body or "insufficient_quota" in body
                    or "add credits" in body or "billing" in body
                )
                if response.status_code in (401, 403) or billing_exhausted:
                    # Account-level failure: retrying every request is pointless.
                    self.fatal_reason = last_error
                    self.abort.set()
                    break
                if (
                    response.status_code not in (408, 409, 429)
                    and response.status_code < 500
                ):
                    break

            if attempt < self.cfg["max_attempts"] and not self.abort.is_set():
                delay = min(
                    self.cfg["max_retry_delay_seconds"],
                    self.cfg["base_retry_delay_seconds"] * (2 ** (attempt - 1)),
                )
                time.sleep(delay * (0.5 + random.random()))

        return {
            "ok": False,
            "attempt": self.cfg["max_attempts"],
            "parsed": None,
            "raw_response": None,
            "usage": attempt_usages[-1] if attempt_usages else {},
            "attempt_usages": attempt_usages,
            "cost_usd": round(total_cost, 6),
            "elapsed_sec": round(elapsed, 2),
            "error": last_error,
        }

    def process(self, row: dict, prefix: list[dict]) -> dict:
        if self.abort.is_set() or not self.budget.reserve():
            self.abort.set()
            return {"skipped": True}
        try:
            result = self.call(row, prefix)
            self.budget.commit(result["cost_usd"])
        except Exception:
            self.budget.release()
            raise

        record = {
            "run_id": self.run_id,
            "stage_limit": self.limit,
            "image_id": row["image_id"],
            "filename": row["filename"],
            "original_filename": row.get("original_filename", ""),
            "property_globalid": row["property_globalid"],
            "property_guid": row["property_guid"],
            "baseline_score": int(row["baseline_score"]) if row.get("baseline_score") else None,
            "x": float(row["x"]) if row.get("x") else None,
            "y": float(row["y"]) if row.get("y") else None,
            "model": self.cfg["model"],
            "predicted_score": (
                result["parsed"].get("score") if result.get("parsed") else None
            ),
            "predicted_category": (
                result["parsed"].get("category") if result.get("parsed") else None
            ),
            "timestamp": time.time(),
            **result,
        }
        self.append(record)
        return record

    def check_warnings(self) -> None:
        for threshold in self.cfg["budget"]["warning_thresholds_usd"]:
            if self.budget.spent >= threshold and threshold not in self.warning_emitted:
                self.warning_emitted.add(threshold)
                print(f"WARNING: spend passed ${threshold:.2f}", flush=True)

    def dry_run(self) -> None:
        done = self.successful_ids()
        pending = [row for row in self.rows if row["image_id"] not in done]
        image_dir = Path(self.rows[0]["image_path"]).parent
        on_disk = {path.name for path in image_dir.iterdir()}
        missing = [row["filename"] for row in self.rows if row["filename"] not in on_disk]
        print(f"DRY RUN run_id={self.run_id}")
        print(f"manifest={self.manifest_path} rows_in_stage={len(self.rows)}")
        print(f"already_done={len(done & {r['image_id'] for r in self.rows})} pending={len(pending)}")
        print(f"missing_files={len(missing)} unique_parcels={len({r['property_globalid'] for r in self.rows})}")
        print(f"model={self.cfg['model']} workers={self.cfg['workers']}")
        print(f"hard_ceiling=${self.cfg['budget']['hard_ceiling_usd']:.2f}")
        print(f"prior_spend=${self.budget.spent:.4f}")
        print("No API calls made.")

    def preflight(self) -> None:
        file_ids = self.ensure_example_files()
        prefix = self.stable_prefix(file_ids)
        row = self.rows[0]
        response = requests.post(
            INPUT_TOKENS_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": self.cfg["model"], "input": self.input_items(row, prefix)},
            timeout=180,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Token preflight failed: HTTP {response.status_code} {response.text[:300]}"
            )
        print("PREFLIGHT OK")
        print(f"example_file_ids={list(file_ids.values())}")
        print(f"input_tokens_first_target={response.json().get('input_tokens')}")
        print("No inference call made.")

    def run(self) -> None:
        if not API_KEY:
            raise SystemExit(f"OPENAI_API_KEY missing in {HERE / '.env'}")
        file_ids = self.ensure_example_files()
        prefix = self.stable_prefix(file_ids)
        done = self.successful_ids()
        pending = [row for row in self.rows if row["image_id"] not in done]
        print(
            f"Terra stage limit={self.limit} done={len(self.rows)-len(pending)} "
            f"pending={len(pending)} prior_spend=${self.budget.spent:.2f}",
            flush=True,
        )
        if not pending:
            self.summarize(0.0)
            return

        started = time.monotonic()
        completed = 0

        # One sequential request warms the explicit prefix cache.
        first = self.process(pending.pop(0), prefix)
        completed += 1
        if not first.get("ok"):
            raise SystemExit(f"Cache warm-up failed: {first.get('error')}")
        print(
            f"cache warm-up: write={first['usage'].get('cache_write_tokens', 0)} "
            f"cached={first['usage'].get('cached_tokens', 0)} "
            f"spent=${self.budget.spent:.3f}",
            flush=True,
        )

        iterator = iter(pending)
        max_pending = self.cfg["max_pending_requests"]
        with ThreadPoolExecutor(max_workers=self.cfg["workers"]) as pool:
            futures = {}
            for _ in range(min(max_pending, len(pending))):
                row = next(iterator, None)
                if row is not None:
                    futures[pool.submit(self.process, row, prefix)] = row

            while futures and not self.abort.is_set():
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    futures.pop(future)
                    result = future.result()
                    if not result.get("skipped"):
                        completed += 1
                    row = next(iterator, None)
                    if row is not None and not self.abort.is_set():
                        futures[pool.submit(self.process, row, prefix)] = row
                    if completed % 50 == 0 or completed == len(pending) + 1:
                        elapsed = time.monotonic() - started
                        rate = completed / max(elapsed, 1e-9)
                        eta = (len(pending) + 1 - completed) / max(rate, 1e-9)
                        print(
                            f"[{completed:>5}/{len(pending)+1}] "
                            f"spent=${self.budget.spent:.2f} "
                            f"rate={rate:.2f}/s eta={eta/60:.1f}m",
                            flush=True,
                        )
                        self.check_warnings()

        if self.fatal_reason:
            raise SystemExit(f"Fatal provider error: {self.fatal_reason}")
        if self.abort.is_set() and self.budget.spent >= self.budget.ceiling:
            raise SystemExit(
                f"Hard budget ceiling reached: ${self.budget.spent:.2f}"
            )
        self.summarize(time.monotonic() - started)

    def summarize(self, elapsed: float) -> dict:
        all_records = self.all_records()
        latest_success = {}
        for record in all_records:
            if record.get("ok"):
                latest_success[record["image_id"]] = record
        stage_ids = {row["image_id"] for row in self.rows}
        records = [
            record
            for image_id, record in latest_success.items()
            if image_id in stage_ids
        ]
        unresolved = len(stage_ids - set(latest_success))

        pairs = [
            (record["predicted_score"], record["baseline_score"])
            for record in records
            if record["baseline_score"] in (1, 2, 3, 4, 5)
        ]
        n = len(pairs)
        distribution = Counter(record["predicted_score"] for record in records)
        total_prompt = sum(r["usage"].get("prompt_tokens", 0) for r in records)
        total_cached = sum(r["usage"].get("cached_tokens", 0) for r in records)
        stage_cost = sum(
            record.get("cost_usd", 0)
            for record in all_records
            if record.get("image_id") in stage_ids
        )
        summary = {
            "run_id": self.run_id,
            "stage_limit": self.limit,
            "successful": len(records),
            "unresolved": unresolved,
            "elapsed_seconds_this_invocation": round(elapsed, 1),
            "throughput_images_per_hour": (
                round(len(records) / max(elapsed / 3600, 1e-9), 1)
                if elapsed
                else None
            ),
            "campaign_spend_usd": round(self.budget.spent, 4),
            "stage_records_cost_usd": round(stage_cost, 4),
            "cost_per_1000_usd": round(
                1000 * stage_cost / max(len(records), 1), 2
            ),
            "cache_hit_rate": round(total_cached / max(total_prompt, 1), 4),
            "mean_latency_seconds": round(
                sum(r["elapsed_sec"] for r in records) / max(len(records), 1), 2
            ),
            "score_distribution": dict(sorted(distribution.items())),
            "vs_baseline": {
                "n": n,
                "exact_accuracy_pct": round(
                    100 * sum(pred == truth for pred, truth in pairs) / max(n, 1),
                    1,
                ),
                "three_class_accuracy_pct": round(
                    100
                    * sum(class3(pred) == class3(truth) for pred, truth in pairs)
                    / max(n, 1),
                    1,
                ),
                "within_1_pct": round(
                    100
                    * sum(abs(pred - truth) <= 1 for pred, truth in pairs)
                    / max(n, 1),
                    1,
                ),
            },
        }
        path = self.run_dir / f"summary_{self.limit}.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n" + json.dumps(summary, indent=2), flush=True)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(HERE / "terra_config.yaml")
    )
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    campaign = Campaign(Path(args.config), args.limit)
    allowed = campaign.cfg["stages"]
    if args.limit not in allowed:
        raise SystemExit(f"--limit must be one of configured stages: {allowed}")
    if args.dry_run:
        campaign.dry_run()
    elif args.preflight:
        campaign.preflight()
    else:
        campaign.run()


if __name__ == "__main__":
    main()
