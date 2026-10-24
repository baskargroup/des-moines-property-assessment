"""
Image few-shot prompt-selection, all three models on the original 100-image
benchmark. Each "[IMAGE: <file> sent as base64]" marker in a prompt file is
replaced IN POSITION by the actual example image; "[TARGET IMAGE ...]" by the
target. Zero-shot prompts (P1-P3) have no markers -> target appended at end.

Models (faithful to src harness params):
  - gpt-4.1          : OpenAI, max_tokens=8000, temperature=0.0
  - gpt-5.2          : OpenAI, max_completion_tokens=8000, reasoning_effort=low
  - gpt-5.6-sol      : OpenAI Responses API, explicit P6 cache breakpoint, reasoning=low
  - gpt-5.6-terra    : OpenAI Responses API, explicit P6 cache breakpoint, reasoning=low
  - gpt-5.6-luna     : OpenAI Responses API, explicit P6 cache breakpoint, reasoning=low
  - gpt-6-astra      : OpenAI, same params as gpt-5.2 (reasoning cannot be disabled)
  - gemini-3-flash   : google-genai, max_output_tokens=8000, temp=0.0, top_p=0.95, top_k=40
  - gemini-3.5-flash : google-genai, same params as gemini-3-flash
  - gemini-3.8-flash : google-genai, same params as gemini-3-flash

Resume-safe (per model/prompt results.csv). Records token usage per call so
cost is measured, not estimated. Prints a combined table (5-class / 3-class
accuracy, MAE, within-1). GPT-4.1 reuses the earlier image-based run.

Usage:
  python run_promptsel_images.py gpt-6-astra gemini-3.8-flash --configs 06 --ceiling 20
"""
import os, sys, csv, json, base64, time, re, argparse, threading, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX_DIR = os.path.join(ROOT, "data", "download", "example_images")
load_dotenv(os.path.join(ROOT, ".env"))  # keys may also come from the environment
if (os.getenv("OPENAI_API_KEY") or "").strip().lower() in ("", "none") and os.getenv("DSM_ENV_FILE"):
    load_dotenv(os.getenv("DSM_ENV_FILE"), override=True)

MAX_TOKENS = 8000
TEMP = 0.0
TOP_P = 0.95
TOP_K = 40

# USD per 1M tokens, used only for the running-cost ceiling and the summary.
PRICES = {
    "gpt-4.1":          {"input": 2.00,  "cached": 0.50,  "cache_write": 2.00,  "output": 8.0},
    "gpt-5.2":          {"input": 1.75,  "cached": 0.175, "cache_write": 1.75,  "output": 14.0},
    "gpt-5.6-sol":      {"input": 4.00,  "cached": 0.40,  "cache_write": 5.00,  "output": 20.0},
    "gpt-5.6-terra":    {"input": 2.00,  "cached": 0.20,  "cache_write": 2.50,  "output": 12.0},
    "gpt-5.6-luna":     {"input": 0.20,  "cached": 0.02,  "cache_write": 0.25,  "output": 1.2},
    "gpt-6-astra":      {"input": 10.00, "cached": 1.00,  "cache_write": 12.50, "output": 50.0},
    "gemini-3-flash":   {"input": 0.50,  "cached": 0.05,  "cache_write": 0.50,  "output": 3.0},
    "gemini-3.5-flash": {"input": 1.50,  "cached": 0.15,  "cache_write": 1.50,  "output": 9.0},
    "gemini-3.8-flash": {"input": 0.75,  "cached": 0.075, "cache_write": 0.75,  "output": 3.75},
}
USAGE_FIELDS = ["prompt_tokens", "cached_tokens", "cache_write_tokens",
                "output_tokens", "reasoning_tokens", "cost_usd"]

CONFIGS = [
    "config_01_zeroshot_original.json",
    "config_02_zeroshot_improved.json",
    "config_03_zeroshot_text_examples.json",
    "config_04_fewshot_1img.json",
    "config_05_fewshot_2img.json",
    "config_06_fewshot_5img.json",
    "config_07_fewshot_8img.json",
]

# out_dir=None -> use OUT_ROOT/<key>; gpt-4.1 reuses the earlier run location.
MODELS = {
    "gpt-4.1":        {"backend": "openai",       "model_id": "gpt-4.1",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "gpt41_promptsel"),
                        "workers": 8},
    "gpt-5.2":        {"backend": "openai_gpt5",   "model_id": "gpt-5.2-2025-12-11",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "promptsel_images", "gpt-5.2"),
                        "workers": 8},
    "gpt-5.6-sol":    {"backend": "openai_responses", "model_id": "gpt-5.6-sol",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "promptsel_images", "gpt-5.6-sol"),
                        "workers": 6},
    "gpt-5.6-terra":  {"backend": "openai_responses", "model_id": "gpt-5.6-terra",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "promptsel_images", "gpt-5.6-terra"),
                        "workers": 8},
    "gpt-5.6-luna":   {"backend": "openai_responses", "model_id": "gpt-5.6-luna",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "promptsel_images", "gpt-5.6-luna"),
                        "workers": 10},
    "gemini-3-flash": {"backend": "google",        "model_id": "gemini-3-flash-preview",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "promptsel_images", "gemini-3-flash"),
                        "workers": 8},
    "gemini-3.5-flash": {"backend": "google",      "model_id": "gemini-3.5-flash",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "promptsel_images", "gemini-3.5-flash"),
                        "workers": 6},
    "gpt-6-astra":    {"backend": "openai_gpt5",   "model_id": "gpt-6-astra",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "promptsel_images", "gpt-6-astra"),
                        "workers": 6},
    "gemini-3.8-flash": {"backend": "google",      "model_id": "gemini-3.8-flash",
                        "out_dir": os.path.join(ROOT, "data", "outputs", "promptsel_images", "gemini-3.8-flash"),
                        "workers": 6},
}

MARKER = re.compile(r"\[IMAGE:\s*(?P<f>[^\]]+?)\s+sent as base64\]|\[TARGET IMAGE sent as base64\]")

_openai_client = None
_g_client = None
_g_types = None


def oai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def gclient():
    global _g_client, _g_types
    if _g_client is None:
        from google import genai
        from google.genai import types
        _g_types = types
        _g_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"),
                                 http_options={"api_version": "v1alpha"})
    return _g_client, _g_types


def mime_of(path):
    return "image/png" if os.path.splitext(path)[1].lower() == ".png" else "image/jpeg"


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def markers(prompt_text):
    """Yield ('text', str) | ('img', filepath) | ('target', None) in order,
    with a trailing ('target', None) if no explicit target marker."""
    out, last, has_target = [], 0, False
    for m in MARKER.finditer(prompt_text):
        seg = prompt_text[last:m.start()]
        if seg.strip():
            out.append(("text", seg))
        if m.group("f"):
            out.append(("img", os.path.join(EX_DIR, m.group("f").strip())))
        else:
            out.append(("target", None)); has_target = True
        last = m.end()
    tail = prompt_text[last:]
    if tail.strip():
        out.append(("text", tail))
    if not has_target:
        out.append(("target", None))
    return out


# ---------- OpenAI-family ----------
def openai_content(seq, target_path):
    parts = []
    for kind, val in seq:
        if kind == "text":
            parts.append({"type": "text", "text": val})
        else:
            path = val if kind == "img" else target_path
            b = base64.b64encode(read_bytes(path)).decode()
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{mime_of(path)};base64,{b}"}})
    return parts


def empty_usage():
    return {k: 0 for k in USAGE_FIELDS}


def call_openai(model_id, seq, target_path, gpt5, cache_key, retries=3):
    content = openai_content(seq, target_path)
    params = {"model": model_id, "messages": [{"role": "user", "content": content}],
              # Prompt caching: every image shares this prompt's prefix (rubric + example
              # images); the key routes all 100 requests to the same server-side cache.
              "extra_body": {"prompt_cache_key": cache_key}}
    if gpt5:
        params["max_completion_tokens"] = MAX_TOKENS
        params["reasoning_effort"] = "low"  # reasoning models only accept default temperature
    else:
        params["max_tokens"] = MAX_TOKENS
        params["temperature"] = TEMP
    for a in range(retries):
        try:
            r = oai().chat.completions.create(**params)
            u = r.usage
            ptd = getattr(u, "prompt_tokens_details", None)
            ctd = getattr(u, "completion_tokens_details", None)
            reasoning = getattr(ctd, "reasoning_tokens", 0) or 0
            usage = {"prompt_tokens": u.prompt_tokens,
                     "cached_tokens": getattr(ptd, "cached_tokens", 0) or 0,
                     "output_tokens": u.completion_tokens - reasoning,  # visible only
                     "reasoning_tokens": reasoning}
            return r.choices[0].message.content, None, usage
        except Exception as e:
            if a < retries - 1:
                time.sleep(5 * (a + 1)); continue
            return None, str(e)[:150], empty_usage()
    return None, "retries_exhausted", empty_usage()


# ---------- OpenAI Responses API (GPT-5.6 explicit caching) ----------
def responses_input(seq, target_path):
    """Convert the existing in-position P6 sequence to Responses content.

    Everything before the target marker is stable. The final stable input_text
    receives an explicit breakpoint; target image and trailing instruction stay
    outside the cached prefix.
    """
    target_idx = next(i for i, (kind, _) in enumerate(seq) if kind == "target")
    stable, changing = seq[:target_idx], seq[target_idx:]
    parts = []
    for kind, val in stable:
        if kind == "text":
            parts.append({"type": "input_text", "text": val})
        else:
            b = base64.b64encode(read_bytes(val)).decode()
            parts.append({"type": "input_image",
                          "image_url": f"data:{mime_of(val)};base64,{b}",
                          "detail": "auto"})
    # P6 has a text segment directly before [TARGET IMAGE], so this does not
    # alter the benchmark prompt; it only marks the end of reusable content.
    for part in reversed(parts):
        if part["type"] == "input_text":
            part["prompt_cache_breakpoint"] = {"mode": "explicit"}
            break
    for kind, val in changing:
        if kind == "text":
            parts.append({"type": "input_text", "text": val})
        else:
            path = val if kind == "img" else target_path
            b = base64.b64encode(read_bytes(path)).decode()
            parts.append({"type": "input_image",
                          "image_url": f"data:{mime_of(path)};base64,{b}",
                          "detail": "auto"})
    return [{"role": "user", "content": parts}]


def response_output_text(data):
    return "".join(
        part.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def call_openai_responses(model_id, seq, target_path, cache_key, retries=3):
    body = {
        "model": model_id,
        "input": responses_input(seq, target_path),
        "max_output_tokens": MAX_TOKENS,
        "reasoning": {"effort": "low"},
        "prompt_cache_key": cache_key,
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
    }
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
               "Content-Type": "application/json"}
    for a in range(retries):
        try:
            r = requests.post("https://api.openai.com/v1/responses",
                              headers=headers, json=body, timeout=180)
            if r.status_code == 200:
                data = r.json()
                usage_raw = data.get("usage", {})
                inp = usage_raw.get("input_tokens_details", {})
                out = usage_raw.get("output_tokens_details", {})
                reasoning = out.get("reasoning_tokens", 0) or 0
                usage = {
                    "prompt_tokens": usage_raw.get("input_tokens", 0) or 0,
                    "cached_tokens": inp.get("cached_tokens", 0) or 0,
                    "cache_write_tokens": inp.get("cache_write_tokens", 0) or 0,
                    "output_tokens": max((usage_raw.get("output_tokens", 0) or 0) - reasoning, 0),
                    "reasoning_tokens": reasoning,
                }
                raw = response_output_text(data)
                if raw:
                    return raw, None, usage
                err = "empty response output"
            else:
                err = f"http_{r.status_code}: {r.text[:250]}"
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    return None, err, empty_usage()
        except requests.RequestException as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        if a < retries - 1:
            time.sleep(5 * (2 ** a))
    return None, err, empty_usage()


# ---------- Google Gemini ----------
def gemini_content(seq, target_path, types):
    parts = []
    for kind, val in seq:
        if kind == "text":
            parts.append(types.Part(text=val))
        else:
            path = val if kind == "img" else target_path
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime_of(path),
                                                            data=read_bytes(path))))
    return parts


def gemini_gen_config(types):
    for name in ("GenerateContentConfig", "GenerationConfig"):
        if hasattr(types, name):
            try:
                return getattr(types, name)(max_output_tokens=MAX_TOKENS, temperature=TEMP,
                                            top_p=TOP_P, top_k=TOP_K)
            except Exception:
                pass
    return None


def call_gemini(model_id, seq, target_path, retries=4):
    client, types = gclient()
    parts = gemini_content(seq, target_path, types)
    cfg = gemini_gen_config(types)
    for a in range(retries):
        try:
            kw = {"model": model_id, "contents": parts}
            if cfg is not None:
                kw["config"] = cfg
            r = client.models.generate_content(**kw)
            um = getattr(r, "usage_metadata", None)
            usage = {"prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                     "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0,
                     "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                     "reasoning_tokens": getattr(um, "thoughts_token_count", 0) or 0}
            return r.text, None, usage
        except Exception as e:
            es = str(e).lower()
            if ("429" in es or "rate" in es or "quota" in es or "resource_exhausted" in es) and a < retries - 1:
                m = re.search(r"retry in ([\d.]+)s", str(e), re.IGNORECASE) or re.search(r"seconds:\s*([\d.]+)", str(e))
                wait = (float(m.group(1)) + 1) if m else 5 * (2 ** a)
                time.sleep(min(wait, 60)); continue
            if a < retries - 1:
                time.sleep(3 * (a + 1)); continue
            return None, str(e)[:150], empty_usage()
    return None, "retries_exhausted", empty_usage()


def cost_usd(mkey, usage):
    p = PRICES[mkey]
    cached = usage["cached_tokens"]
    writes = usage.get("cache_write_tokens", 0)
    uncached = max(usage["prompt_tokens"] - cached - writes, 0)
    out = usage["output_tokens"] + usage["reasoning_tokens"]
    return (uncached * p["input"] + cached * p["cached"]
            + writes * p["cache_write"] + out * p["output"]) / 1e6


# ---------- shared ----------
def load_gt(gt_file):
    gt = {}
    with open(gt_file, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            gt[r["filename"]] = {"id": r["id"], "ground_truth": int(r["ground_truth"])}
    return gt


def parse_score(raw):
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    if not s.startswith("{"):
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j > i:
            s = s[i:j + 1]
    try:
        return json.loads(s).get("score")
    except Exception:
        return None


def as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def c3(s):
    return "H" if s in (1, 2) else "I" if s == 3 else "U" if s in (4, 5) else None


class Budget:
    """Thread-safe running cost with a hard USD ceiling per model."""

    def __init__(self, ceiling):
        self.ceiling = ceiling
        self.spent = 0.0
        self.lock = threading.Lock()
        self.tripped = False

    def allows(self):
        with self.lock:
            return self.ceiling is None or self.spent < self.ceiling

    def add(self, usd):
        with self.lock:
            self.spent += usd
            if self.ceiling is not None and self.spent >= self.ceiling and not self.tripped:
                self.tripped = True
                print(f"  !! ceiling ${self.ceiling:.2f} reached (spent ${self.spent:.2f}); "
                      f"remaining calls skipped", flush=True)


def run_model_config(mkey, mcfg, cfg_name, budget):
    cfg = json.load(open(os.path.join(ROOT, "experiments", cfg_name)))
    exp = cfg["experiment_name"]
    prompt = open(os.path.join(ROOT, cfg["prompt_file"]), encoding="utf-8").read()
    seq = markers(prompt)
    n_img = sum(1 for k, _ in seq if k == "img")
    gt = load_gt(os.path.join(ROOT, cfg["ground_truth_file"]))
    images_dir = os.path.join(ROOT, cfg["images_dir"])
    out_dir = os.path.join(mcfg["out_dir"], exp)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")

    done = {}
    if os.path.exists(out_csv):
        for r in csv.DictReader(open(out_csv, encoding="utf-8")):
            done[r["filename"]] = r
    todo = [(fn, g) for fn, g in gt.items()
            if os.path.exists(os.path.join(images_dir, fn)) and fn not in done]
    print(f"[{mkey}/{exp}] ex_imgs={n_img} total={len(gt)} done={len(done)} todo={len(todo)}", flush=True)
    results = list(done.values())

    backend = mcfg["backend"]
    model_id = mcfg["model_id"]

    cache_key = f"promptsel-{exp}"

    def work(item):
        fn, g = item
        tp = os.path.join(images_dir, fn)
        if not budget.allows():
            return None
        if backend == "google":
            raw, err, usage = call_gemini(model_id, seq, tp)
        elif backend == "openai_responses":
            raw, err, usage = call_openai_responses(model_id, seq, tp, cache_key=cache_key)
        else:
            raw, err, usage = call_openai(model_id, seq, tp, gpt5=(backend == "openai_gpt5"),
                                          cache_key=cache_key)
        usage["cost_usd"] = round(cost_usd(mkey, usage), 6)
        budget.add(usage["cost_usd"])
        return {"id": g["id"], "filename": fn, "ground_truth": g["ground_truth"],
                "predicted_score": parse_score(raw), "success": raw is not None, "error": err,
                **usage}

    if todo:
        t0 = time.time()
        # GPT-5.6 explicit caching: make exactly one cache-writing request
        # before fan-out, so the remaining 99 are cache reads rather than
        # racing to create duplicate writes.
        if backend == "openai_responses":
            first = work(todo.pop(0))
            if first is not None:
                results.append(first)
                if not first["success"]:
                    raise SystemExit(
                        f"[{mkey}/{exp}] cache warm-up failed: {first['error']}"
                    )
            print(f"  [{mkey}/{exp}] cache warm-up complete  spent ${budget.spent:.2f}", flush=True)
        with ThreadPoolExecutor(max_workers=mcfg["workers"]) as ex:
            futs = {ex.submit(work, it): it for it in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is not None:
                    results.append(r)
                if i % 20 == 0 or i == len(todo):
                    print(f"  [{mkey}/{exp}] {i}/{len(todo)}  spent ${budget.spent:.2f}", flush=True)
        print(f"  [{mkey}/{exp}] {len(todo)} calls in {time.time() - t0:.0f}s", flush=True)
        # Only successful rows are persisted, so failures are retried on the next run.
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "filename", "ground_truth", "predicted_score",
                                              "success", "error"] + USAGE_FIELDS)
            w.writeheader()
            for r in results:
                if str(r.get("success")).lower() == "true":
                    w.writerow({k: r.get(k) for k in w.fieldnames})
    return results


def score(results):
    n = len(results)
    pairs = [(as_int(r["predicted_score"]), int(r["ground_truth"])) for r in results]
    e5 = sum(1 for p, g in pairs if p == g)
    e3 = sum(1 for p, g in pairs if c3(p) == c3(g))
    parsed = [(p, g) for p, g in pairs if p is not None]
    within1 = sum(1 for p, g in parsed if abs(p - g) <= 1)
    mae = (sum(abs(p - g) for p, g in parsed) / len(parsed)) if parsed else None
    pct = lambda k: round(100 * k / n, 1) if n else 0
    return n, len(parsed), pct(e5), pct(e3), pct(within1), (round(mae, 3) if mae is not None else None)


def usage_summary(rows, mkey):
    """Mean tokens per call and total cost, when the rows carry usage columns."""
    with_usage = [r for r in rows if as_int(r.get("prompt_tokens"))]
    if not with_usage:
        return None
    n = len(with_usage)
    mean = lambda k: round(sum(as_int(r.get(k)) or 0 for r in with_usage) / n)
    cost = sum(float(r.get("cost_usd") or 0) for r in with_usage)
    return {"n": n, "prompt_tokens": mean("prompt_tokens"), "cached_tokens": mean("cached_tokens"),
            "cache_write_tokens": mean("cache_write_tokens"),
            "output_tokens": mean("output_tokens"), "reasoning_tokens": mean("reasoning_tokens"),
            "total_cost_usd": round(cost, 4), "cost_per_100_usd": round(100 * cost / n, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["gpt-5.2", "gemini-3-flash"])
    ap.add_argument("--configs", nargs="*", default=None,
                    help="substrings selecting config files, e.g. --configs 06 (default: all)")
    ap.add_argument("--ceiling", type=float, default=None, help="hard USD ceiling per model")
    args = ap.parse_args()
    configs = [c for c in CONFIGS if not args.configs or any(s in c for s in args.configs)]
    if not configs:
        sys.exit(f"no configs match {args.configs}")

    for mkey in args.models:
        mcfg = MODELS[mkey]
        budget = Budget(args.ceiling)
        for cfg_name in configs:
            run_model_config(mkey, mcfg, cfg_name, budget)
        print(f"[{mkey}] total spent this run: ${budget.spent:.2f}", flush=True)

    # combined table across every model that has results on disk
    all_models = list(MODELS)
    combined, usage = {}, {}
    for cfg_name in CONFIGS:
        exp = json.load(open(os.path.join(ROOT, "experiments", cfg_name)))["experiment_name"]
        combined[exp] = {}
        for mkey in all_models:
            csvp = os.path.join(MODELS[mkey]["out_dir"], exp, "results.csv")
            if os.path.exists(csvp):
                rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
                combined[exp][mkey] = score(rows)
                u = usage_summary(rows, mkey)
                if u:
                    usage.setdefault(exp, {})[mkey] = u

    print("\n" + "=" * 110)
    print("IMAGE FEW-SHOT PROMPT-SELECTION (original 100)  |  5-class % / 3-class % / within-1 % / MAE")
    print("=" * 110)
    print(f"{'Prompt':<28}" + "".join(f"{m:>20}" for m in all_models))
    for exp, d in combined.items():
        row = f"{exp:<28}"
        for mkey in all_models:
            v = d.get(mkey)
            row += f"{(f'{v[2]}/{v[3]}/{v[4]}/{v[5]}') if v else '-':>20}"
        print(row)
    if usage:
        print("\nMeasured usage (mean tokens per call; cost per 100 images):")
        for exp, d in usage.items():
            for mkey, u in d.items():
                print(f"  {exp:<26} {mkey:<18} prompt={u['prompt_tokens']:>6} cached={u['cached_tokens']:>6} "
                      f"write={u['cache_write_tokens']:>6} "
                      f"out={u['output_tokens']:>5} reasoning={u['reasoning_tokens']:>5}  ${u['cost_per_100_usd']:.2f}/100")
    out = os.path.join(ROOT, "data", "outputs", "promptsel_images", "combined_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"accuracy": combined, "usage": usage}, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
