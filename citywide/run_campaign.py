"""Three-model P6 scoring campaign runner (GPT-5.2, Gemini 3 Flash, GPT-4.1).

Design (adapted from src/providers/api_request_parallel.py in the research repo):
  - per-provider concurrency cap + requests-per-minute token bucket
  - retry with exponential backoff + jitter; provider-wide cooldown after a 429
  - every (image, model) result is appended to predictions.jsonl immediately,
    so a crash or Ctrl-C loses nothing and re-running resumes where it stopped
  - hard dollar ceiling computed from *real* usage reported by each provider

Prompt caching:
  - OpenAI: the request prefix (rubric text + 5 example images) is byte-identical
    on every call, so the server caches it. `prompt_cache_key` routes all calls to
    the same cache; `prompt_cache_retention="24h"` keeps it warm for the run.
    Cached tokens are billed at the cached-input rate and reported back in
    usage.prompt_tokens_details.cached_tokens.
  - Gemini: the same prefix is uploaded once as an explicit `cachedContents`
    object; each request sends only the target image plus a reference to it.
    Cached tokens are reported in usageMetadata.cachedContentTokenCount.

Usage:
  python run_campaign.py --dry-run                 # no API calls; prints plan
  python run_campaign.py                           # runs config run_id/manifest
  python run_campaign.py --limit 20 --ceiling 2    # small test
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=True)  # campaign keys only; never the app's .env

TAIL_TEXT = "\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
SAFETY_OFF = [{"category": c, "threshold": "BLOCK_NONE"} for c in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]


# ----------------------------------------------------------------------------
# Prompt prefix (identical bytes on every request -> cacheable)
# ----------------------------------------------------------------------------

def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def mime_for(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


class Prefix:
    """Rubric text + 5 example (description, image) pairs, pre-encoded once."""

    def __init__(self, module_dir: Path):
        sys.path.insert(0, str(module_dir))
        cwd = os.getcwd()
        os.chdir(module_dir)  # prompt.py resolves example paths relative to its dir
        try:
            from prompt import get_prompt_and_examples
            text, examples = get_prompt_and_examples()
            self.text = text
            self.examples = []
            for path, score, desc in examples:
                p = Path(path).resolve()
                self.examples.append((f"\n--- EXAMPLE (Score {score}) ---\n{desc}",
                                      mime_for(p), b64_file(p), p.name))
        finally:
            os.chdir(cwd)
        h = hashlib.sha256(self.text.encode())
        for t, m, b, _ in self.examples:
            h.update(t.encode()); h.update(m.encode()); h.update(b.encode())
        self.sha256 = h.hexdigest()

    def openai_parts(self) -> list[dict]:
        parts = [{"type": "text", "text": self.text}]
        for t, m, b, _ in self.examples:
            parts.append({"type": "text", "text": t})
            parts.append({"type": "image_url", "image_url": {"url": f"data:{m};base64,{b}"}})
        return parts

    def gemini_parts(self) -> list[dict]:
        parts = [{"text": self.text}]
        for t, m, b, _ in self.examples:
            parts.append({"text": t})
            parts.append({"inline_data": {"mime_type": m, "data": b}})
        return parts


# ----------------------------------------------------------------------------
# Throttling and retry helpers
# ----------------------------------------------------------------------------

class RateLimiter:
    """Requests-per-minute token bucket shared by all calls to one provider."""

    def __init__(self, rpm: float):
        self.rpm = rpm
        self.capacity = rpm
        self.updated = time.monotonic()
        self.cooldown_until = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self.lock:
                now = time.monotonic()
                self.capacity = min(self.rpm, self.capacity + (now - self.updated) * self.rpm / 60)
                self.updated = now
                wait_cooldown = self.cooldown_until - now
                if wait_cooldown <= 0 and self.capacity >= 1:
                    self.capacity -= 1
                    return
            await asyncio.sleep(max(wait_cooldown, 0.25))

    def trip_cooldown(self, seconds: float):
        self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)


class RetryableError(Exception):
    """Transient failure (429, 5xx, timeout, network, empty response)."""


class CacheExpiredError(RetryableError):
    """Gemini cachedContents reference no longer valid; recreate then retry."""


def backoff(attempt: int, base: float, cap: float) -> float:
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random())


def parse_json(raw: str | None):
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
            s = s[i:j + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------------

class OpenAIProvider:
    def __init__(self, cfg: dict, prefix: Prefix, retry: dict):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                  timeout=retry["request_timeout_seconds"], max_retries=0)
        self.cfg = cfg
        self.prefix_parts = prefix.openai_parts()
        self.limiter = RateLimiter(cfg["requests_per_minute"])
        self.sem = asyncio.Semaphore(cfg["max_concurrent"])
        self.retry = retry

    def build_params(self, mcfg: dict, target_b64: str) -> dict:
        content = list(self.prefix_parts)
        content.append({"type": "text", "text": TAIL_TEXT})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{target_b64}"}})
        params = {
            "model": mcfg["model"],
            "messages": [{"role": "user", "content": content}],
            # --- prompt caching controls ---
            "extra_body": {
                "prompt_cache_key": self.cfg["prompt_cache_key"],
                "prompt_cache_retention": self.cfg["prompt_cache_retention"],
            },
        }
        if "reasoning_effort" in mcfg:
            params["max_completion_tokens"] = mcfg["max_output_tokens"]
            params["reasoning_effort"] = mcfg["reasoning_effort"]
        else:
            params["max_tokens"] = mcfg["max_output_tokens"]
            params["temperature"] = mcfg.get("temperature", 0.0)
        return params

    async def call(self, mcfg: dict, target_b64: str) -> dict:
        """Caller must hold self.sem (see Campaign.score_one)."""
        await self.limiter.acquire()
        t0 = time.monotonic()
        try:
            resp = await self.client.chat.completions.create(**self.build_params(mcfg, target_b64))
        except Exception as e:  # classify below
            status = getattr(e, "status_code", None)
            if status == 429:
                self.limiter.trip_cooldown(self.retry["cooldown_after_429_seconds"])
            if status in (408, 409, 429) or (status is not None and status >= 500) or status is None:
                raise RetryableError(f"{type(e).__name__} {status}: {str(e)[:200]}") from e
            raise
        raw = resp.choices[0].message.content
        if not raw:
            raise RetryableError("empty completion")
        u = resp.usage
        ptd = getattr(u, "prompt_tokens_details", None)
        ctd = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(ctd, "reasoning_tokens", 0) or 0
        return {
            "raw_response": raw,
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "usage": {
                "prompt_tokens": u.prompt_tokens,
                "cached_tokens": getattr(ptd, "cached_tokens", 0) or 0,
                # completion_tokens already includes reasoning; store visible only so
                # cost_usd (= output + reasoning) does not double count.
                "output_tokens": u.completion_tokens - reasoning,
                "reasoning_tokens": reasoning,
            },
        }


class GeminiProvider:
    def __init__(self, cfg: dict, prefix: Prefix, retry: dict, session: aiohttp.ClientSession):
        self.key = os.environ["GOOGLE_API_KEY"]
        self.cfg = cfg
        self.prefix = prefix
        self.session = session
        self.limiter = RateLimiter(cfg["requests_per_minute"])
        self.sem = asyncio.Semaphore(cfg["max_concurrent"])
        self.retry = retry
        self.timeout = aiohttp.ClientTimeout(total=retry["request_timeout_seconds"])
        self.cache_name: str | None = None
        self.cache_expires: float = 0.0
        self.cache_tokens: int = 0
        self.cache_lock = asyncio.Lock()
        self.mode = "explicit_cache" if cfg["explicit_cache"] else "inline"

    # --- explicit context cache -------------------------------------------
    async def ensure_cache(self, model: str):
        if self.mode != "explicit_cache":
            return
        async with self.cache_lock:
            margin = self.cfg["cache_refresh_margin_seconds"]
            if self.cache_name and time.time() < self.cache_expires - margin:
                return
            if self.cache_name and time.time() < self.cache_expires:
                await self._extend_cache()
                return
            await self._create_cache(model)

    async def _create_cache(self, model: str):
        body = {
            "model": f"models/{model}",
            "displayName": self.cfg["cache_display_name"],
            "contents": [{"role": "user", "parts": self.prefix.gemini_parts()}],
            "ttl": f"{self.cfg['cache_ttl_seconds']}s",
        }
        async with self.session.post(f"{GEMINI_BASE}/cachedContents?key={self.key}",
                                     json=body, timeout=self.timeout) as r:
            data = await r.json()
            if r.status != 200:
                print(f"[gemini] explicit cache creation failed ({r.status}): "
                      f"{str(data)[:200]} -> falling back to inline prefix", flush=True)
                self.mode = "inline"
                return
        self.cache_name = data["name"]
        self.cache_tokens = int(data.get("usageMetadata", {}).get("totalTokenCount", 0))
        self.cache_expires = time.time() + self.cfg["cache_ttl_seconds"]
        print(f"[gemini] cache {self.cache_name} created: {self.cache_tokens} tokens, "
              f"ttl {self.cfg['cache_ttl_seconds']}s", flush=True)

    async def _extend_cache(self):
        url = f"{GEMINI_BASE}/{self.cache_name}?updateMask=ttl&key={self.key}"
        async with self.session.patch(url, json={"ttl": f"{self.cfg['cache_ttl_seconds']}s"},
                                      timeout=self.timeout) as r:
            if r.status == 200:
                self.cache_expires = time.time() + self.cfg["cache_ttl_seconds"]
                print("[gemini] cache ttl extended", flush=True)
            else:
                self.cache_name = None  # force recreate on next ensure_cache

    async def delete_cache(self):
        if self.cache_name:
            try:
                async with self.session.delete(f"{GEMINI_BASE}/{self.cache_name}?key={self.key}",
                                               timeout=self.timeout) as r:
                    print(f"[gemini] cache deleted ({r.status})", flush=True)
            except Exception as e:
                print(f"[gemini] cache delete failed: {e}", flush=True)

    # --- request -----------------------------------------------------------
    def build_body(self, mcfg: dict, target_b64: str) -> dict:
        parts = [] if self.mode == "explicit_cache" else self.prefix.gemini_parts()
        parts.append({"text": TAIL_TEXT})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": target_b64}})
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": mcfg.get("temperature", 0.0),
                                 "maxOutputTokens": mcfg["max_output_tokens"]},
            "safetySettings": SAFETY_OFF,
        }
        if self.mode == "explicit_cache":
            body["cachedContent"] = self.cache_name
        return body

    async def call(self, mcfg: dict, target_b64: str) -> dict:
        """Caller must hold self.sem (see Campaign.score_one)."""
        await self.ensure_cache(mcfg["model"])
        await self.limiter.acquire()
        url = f"{GEMINI_BASE}/models/{mcfg['model']}:generateContent?key={self.key}"
        t0 = time.monotonic()
        try:
            async with self.session.post(url, json=self.build_body(mcfg, target_b64),
                                         timeout=self.timeout) as r:
                status = r.status
                data = await r.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise RetryableError(f"network: {e}") from e
        if status == 429:
            self.limiter.trip_cooldown(self.retry["cooldown_after_429_seconds"])
            raise RetryableError("429 rate limited")
        if status >= 500:
            raise RetryableError(f"http {status}")
        if status != 200:
            msg = str(data)[:300]
            if self.mode == "explicit_cache" and ("cachedContent" in msg or "CachedContent" in msg):
                self.cache_name = None
                raise CacheExpiredError(msg)
            raise RuntimeError(f"http {status}: {msg}")
        cands = data.get("candidates") or []
        raw = " ".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", [])
                       if "text" in p).strip() if cands else ""
        if not raw:
            raise RetryableError(f"no candidates / empty text: {str(data)[:200]}")
        um = data.get("usageMetadata", {})
        return {
            "raw_response": raw,
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "usage": {
                "prompt_tokens": um.get("promptTokenCount", 0),
                "cached_tokens": um.get("cachedContentTokenCount", 0),
                "output_tokens": um.get("candidatesTokenCount", 0),
                "reasoning_tokens": um.get("thoughtsTokenCount", 0),
            },
        }


# ----------------------------------------------------------------------------
# Campaign
# ----------------------------------------------------------------------------

def cost_usd(usage: dict, prices: dict) -> float:
    cached = usage["cached_tokens"]
    uncached = max(usage["prompt_tokens"] - cached, 0)
    out = usage["output_tokens"] + usage["reasoning_tokens"]
    return (uncached * prices["input"] + cached * prices["cached"] + out * prices["output"]) / 1e6


class Campaign:
    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.run_id = args.run_id or cfg["run_id"]
        self.run_dir = HERE / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.pred_path = self.run_dir / "predictions.jsonl"
        self.ceiling = args.ceiling if args.ceiling is not None else cfg["budget"]["hard_ceiling_usd"]
        self.est_per_req = cfg["budget"]["estimated_cost_per_request_usd"]
        self.prefix = Prefix((HERE / cfg["prompt_module_dir"]).resolve())
        self.manifest = self.load_manifest(HERE / (args.manifest or cfg["manifest"]), args.limit)
        self.models = {k: v for k, v in cfg["models"].items() if not args.models or k in args.models}
        self.spent = self.prior_spend()
        self.in_flight = 0
        self.stop_reason: str | None = None
        self.write_lock = asyncio.Lock()
        self.stats = {k: {"ok": 0, "failed": 0, "attempts": 0} for k in self.models}

    def prior_spend(self) -> float:
        """Return billed spend already recorded for this run and model subset."""
        pred_path = self.run_dir / "predictions.jsonl"
        if not pred_path.exists():
            return 0.0
        total = 0.0
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("run_id") == self.run_id
                    and record.get("model_key") in self.models
                ):
                    total += float(record.get("cost_usd") or 0)
        return total

    @staticmethod
    def load_manifest(path: Path, limit: int | None) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[:limit] if limit else rows

    def done_pairs(self) -> set[tuple[str, str]]:
        done = set()
        if self.pred_path.exists():
            with self.pred_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("ok"):
                        done.add((r["image_id"], r["model_key"]))
        return done

    def pending(self) -> list[tuple[dict, str]]:
        done = self.done_pairs()
        return [(row, mk) for row in self.manifest for mk in self.models
                if (row["image_id"], mk) not in done]

    async def write(self, rec: dict):
        async with self.write_lock:
            with self.pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def budget_allows(self) -> bool:
        return self.spent + (self.in_flight + 1) * self.est_per_req <= self.ceiling

    async def score_one(self, row: dict, mk: str, providers: dict, progress: list):
        mcfg = self.models[mk]
        provider = providers[mcfg["provider"]]
        prices = self.cfg["prices_per_million"][mk]
        retry = self.cfg["retry"]
        # Load the image only after this task acquires a provider slot. Creating
        # all citywide tasks at once must not retain 66,701 base64 images.
        target_b64 = None
        base = {"image_id": row["image_id"], "filename": row["filename"],
                "property_globalid": row["property_globalid"],
                "property_guid": row.get("property_guid", ""),
                "baseline_score": row.get("baseline_score", ""),
                "model_key": mk, "model": mcfg["model"], "run_id": self.run_id}
        last_err = None
        request_cost = 0.0
        for attempt in range(1, retry["max_attempts"] + 1):
            if self.stop_reason:
                return
            # Hold the provider's concurrency slot *before* the budget check so
            # in_flight counts requests actually on the wire, not queued tasks.
            async with provider.sem:
                if self.stop_reason:
                    return
                if not self.budget_allows():
                    self.stop_reason = f"budget ceiling ${self.ceiling:.2f} reached"
                    return
                if target_b64 is None:
                    target_b64 = b64_file(Path(row["image_path"]))
                self.in_flight += 1
                self.stats[mk]["attempts"] += 1
                try:
                    res = await provider.call(mcfg, target_b64)
                    # Provider-reported usage is billable even when the response
                    # cannot be parsed and must be retried.
                    c = cost_usd(res["usage"], prices)
                    request_cost += c
                    self.spent += c
                    parsed = parse_json(res["raw_response"])
                    if parsed is None or not isinstance(parsed.get("score"), int):
                        raise RetryableError("unparseable JSON / missing score")
                except RetryableError as e:
                    last_err = str(e)
                    res = None
                except Exception as e:  # non-retryable (400 bad request, auth, ...)
                    last_err = f"non-retryable {type(e).__name__}: {str(e)[:300]}"
                    res = None
                    attempt = retry["max_attempts"]  # no further attempts
                finally:
                    self.in_flight -= 1
            if res is None:
                if attempt < retry["max_attempts"]:
                    await asyncio.sleep(backoff(attempt, retry["base_delay_seconds"],
                                                retry["max_delay_seconds"]))
                    continue
                break
            rec = {**base, "ok": True, "attempt": attempt, "parsed": parsed,
                   "predicted_score": parsed.get("score"),
                   "predicted_category": parsed.get("category"),
                   "raw_response": res["raw_response"], "usage": res["usage"],
                   "cost_usd": round(request_cost, 6), "elapsed_sec": res["elapsed_sec"],
                   "cache_mode": getattr(provider, "mode", "prompt_cache"),
                   "ts": datetime.now(timezone.utc).isoformat()}
            await self.write(rec)
            self.stats[mk]["ok"] += 1
            progress[0] += 1
            if progress[0] % 10 == 0 or progress[0] == progress[1]:
                print(f"  [{progress[0]}/{progress[1]}] spent ${self.spent:.3f}", flush=True)
            return
        await self.write({**base, "ok": False, "attempt": retry["max_attempts"],
                          "error": last_err, "cost_usd": round(request_cost, 6),
                          "ts": datetime.now(timezone.utc).isoformat()})
        self.stats[mk]["failed"] += 1
        progress[0] += 1

    async def run(self):
        pending = self.pending()
        total_pairs = len(self.manifest) * len(self.models)
        print(f"run_id={self.run_id}  manifest={len(self.manifest)} images  models={list(self.models)}")
        print(f"pairs total={total_pairs}  already done={total_pairs - len(pending)}  pending={len(pending)}")
        print(f"prefix sha256={self.prefix.sha256[:16]}...  ceiling=${self.ceiling:.2f}")
        if not pending:
            print("nothing to do"); return
        t0 = time.monotonic()
        async with aiohttp.ClientSession() as session:
            providers = {
                "openai": OpenAIProvider(self.cfg["openai"], self.prefix, self.cfg["retry"]),
                "gemini": GeminiProvider(self.cfg["gemini"], self.prefix, self.cfg["retry"], session),
            }
            progress = [0, len(pending)]

            # Keep only a bounded number of active coroutines. Creating one task
            # per citywide pair retains tens of thousands of coroutine frames
            # and can exhaust the host when two model campaigns run together.
            pending_iter = iter(pending)

            async def worker():
                while not self.stop_reason:
                    try:
                        row, mk = next(pending_iter)
                    except StopIteration:
                        return
                    await self.score_one(row, mk, providers, progress)

            active_providers = {mcfg["provider"] for mcfg in self.models.values()}
            worker_count = sum(
                int(self.cfg[provider_name]["max_concurrent"])
                for provider_name in active_providers
            )
            try:
                await asyncio.gather(*(worker() for _ in range(worker_count)))
            finally:
                await providers["gemini"].delete_cache()
        self.summarize(time.monotonic() - t0)

    def summarize(self, elapsed: float):
        recs = [json.loads(l) for l in self.pred_path.open(encoding="utf-8") if l.strip()]
        ok = [r for r in recs if r.get("ok") and r["run_id"] == self.run_id]
        per_model = {}
        for mk in self.models:
            rs = [r for r in ok if r["model_key"] == mk]
            if not rs:
                per_model[mk] = {"ok": 0}; continue
            n = len(rs)
            u = lambda k: sum(r["usage"][k] for r in rs) / n
            cached_rate = sum(r["usage"]["cached_tokens"] for r in rs) / max(sum(r["usage"]["prompt_tokens"] for r in rs), 1)
            per_model[mk] = {
                "ok": n, "failed": self.stats[mk]["failed"], "attempts": self.stats[mk]["attempts"],
                "mean_latency_sec": round(sum(r["elapsed_sec"] for r in rs) / n, 2),
                "mean_prompt_tokens": round(u("prompt_tokens")), "mean_cached_tokens": round(u("cached_tokens")),
                "cache_hit_rate": round(cached_rate, 3),
                "mean_output_tokens": round(u("output_tokens")), "mean_reasoning_tokens": round(u("reasoning_tokens")),
                "total_cost_usd": round(sum(r["cost_usd"] for r in rs), 4),
                "cost_per_1000_usd": round(1000 * sum(r["cost_usd"] for r in rs) / n, 2),
            }
        images_done = len({r["image_id"] for r in ok})
        summary = {
            "run_id": self.run_id, "stop_reason": self.stop_reason,
            "elapsed_sec": round(elapsed, 1), "images_scored": images_done,
            "images_per_hour": round(images_done / max(elapsed / 3600, 1e-9), 1),
            "total_cost_usd": round(sum(r["cost_usd"] for r in ok), 4),
            "cost_per_1000_images_all_models_usd": round(1000 * sum(r["cost_usd"] for r in ok) / max(images_done, 1), 2),
            "per_model": per_model, "prefix_sha256": self.prefix.sha256,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n" + json.dumps(summary, indent=2))

    def dry_run(self):
        pending = self.pending()
        print(f"DRY RUN  run_id={self.run_id}  manifest={len(self.manifest)} images  models={list(self.models)}")
        print(f"pending (image, model) pairs: {len(pending)}")
        print(f"prefix sha256: {self.prefix.sha256}")
        print(f"examples: {[e[3] for e in self.prefix.examples]}")
        print(f"budget ceiling: ${self.ceiling:.2f}   est/request ${self.est_per_req}  "
              f"-> max requests under ceiling ~{int(self.ceiling / self.est_per_req)}")
        print(f"openai caching: prompt_cache_key={self.cfg['openai']['prompt_cache_key']} "
              f"retention={self.cfg['openai']['prompt_cache_retention']}")
        print(f"gemini caching: explicit_cache={self.cfg['gemini']['explicit_cache']} "
              f"ttl={self.cfg['gemini']['cache_ttl_seconds']}s")
        missing = [r["image_path"] for r in self.manifest if not Path(r["image_path"]).exists()]
        print(f"missing image files: {len(missing)}")
        print("no API calls made.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--run-id")
    ap.add_argument("--manifest")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--models", nargs="*", help="subset of model keys, e.g. gpt41 gem3fp")
    ap.add_argument("--ceiling", type=float, help="hard USD ceiling override")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true",
                    help="recompute summary.json from predictions.jsonl, no API calls")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    campaign = Campaign(cfg, args)
    if args.dry_run:
        campaign.dry_run()
        return
    if args.summarize_only:
        campaign.summarize(elapsed=0.0)
        return
    for k in ("OPENAI_API_KEY", "GOOGLE_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"{k} missing in {HERE / '.env'}")
    asyncio.run(campaign.run())


if __name__ == "__main__":
    main()
