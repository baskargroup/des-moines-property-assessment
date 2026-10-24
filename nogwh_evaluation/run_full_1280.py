"""
Full NOGWH benchmark: ALL 1,280 images x 3 models with the P6 (5-image few-shot) prompt.

Ensemble:
  - gpt-5.2-2025-12-11      (OpenAI, paper's exact GPT)
  - gemini-3-flash-preview  (Google, paper's exact Gemini)
  - gpt-4.1                 (OpenAI, substitute for inaccessible Qwen3-VL-32B)

Resume-safe: appends to predictions_full.jsonl, skips already-done image_ids.
"""
import os, sys, csv, json, time, base64, asyncio, collections
from pathlib import Path
import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, r".")
from prompt import get_prompt_and_examples

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

prompt_text, examples = get_prompt_and_examples()


def encode(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


# Pre-encode the 5 example images once (reused for every request)
EX_OAI, EX_GEM = [], []
for path, score, desc in examples:
    desc_text = f"\n--- EXAMPLE (Score {score}) ---\n{desc}"
    EX_OAI.append(("text", desc_text))
    EX_GEM.append({"text": desc_text})
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    b = encode(path)
    EX_OAI.append(("image", f"data:{mime};base64,{b}"))
    EX_GEM.append({"inline_data": {"mime_type": mime, "data": b}})


def openai_content(target_path):
    content = [{"type": "text", "text": prompt_text}]
    for kind, value in EX_OAI:
        content.append({"type": "text", "text": value} if kind == "text"
                       else {"type": "image_url", "image_url": {"url": value}})
    content.append({"type": "text", "text": "\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode(target_path)}"}})
    return [{"role": "user", "content": content}]


def gemini_parts(target_path):
    parts = [{"text": prompt_text}]
    parts.extend(EX_GEM)
    parts.append({"text": "\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."})
    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": encode(target_path)}})
    return parts


client = AsyncOpenAI(api_key=OPENAI_KEY)


async def call_openai(model, target):
    is_reasoning = ("gpt-5" in model and "chat-latest" not in model) or model.startswith(("o1","o3","o4"))
    params = {"model": model, "messages": openai_content(target)}
    if is_reasoning:
        params["max_completion_tokens"] = 8000
        params["temperature"] = 1.0
        if any(v in model for v in ["5.2","5.4","5.5","o3","o4"]):
            params["reasoning_effort"] = "low"
    else:
        params["max_tokens"] = 8000
        params["temperature"] = 0.0
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = await client.chat.completions.create(**params)
            return resp.choices[0].message.content, time.time()-t0, None
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                await asyncio.sleep([3, 10, 25][attempt]); continue
            return None, 0, err[:120]
    return None, 0, "retries_exhausted"


async def call_gemini(session, model, target):
    payload = {
        "contents": [{"role": "user", "parts": gemini_parts(target)}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8000},
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in
            ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]],
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_KEY}"
    for attempt in range(5):
        try:
            t0 = time.time()
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as r:
                if r.status == 200:
                    data = await r.json()
                    cands = data.get("candidates", [])
                    if cands:
                        text = " ".join(p.get("text","") for p in cands[0].get("content",{}).get("parts",[]) if "text" in p).strip()
                        if text:
                            return text, time.time()-t0, None
                    return None, time.time()-t0, "no_candidates"
                elif r.status in (429, 503):
                    await asyncio.sleep([3, 10, 30, 60, 90][attempt]); continue
                else:
                    body = await r.text()
                    return None, time.time()-t0, f"http_{r.status} {body[:100]}"
        except Exception:
            await asyncio.sleep([3, 10, 30, 60, 90][attempt])
    return None, 0, "retries_exhausted"


def parse(raw):
    if not raw: return None
    s = raw.strip()
    if "```json" in s: s = s.split("```json",1)[1].split("```",1)[0].strip()
    elif "```" in s:   s = s.split("```",1)[1].split("```",1)[0].strip()
    if not s.startswith("{"):
        i = s.find("{"); j = s.rfind("}")
        if i != -1 and j > i: s = s[i:j+1]
    try: return json.loads(s)
    except: return None


ENSEMBLE = {
    "gpt52":  "gpt-5.2-2025-12-11",
    "gem3fp": "gemini-3-flash-preview",
    "gpt41":  "gpt-4.1",
}

OUT = r"predictions_full.jsonl"
_lock = asyncio.Lock()


async def write_record(rec):
    async with _lock:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_done():
    if not os.path.exists(OUT): return set()
    s = set()
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            try: s.add(json.loads(line)["image_id"])
            except: pass
    return s


async def process(row, session, sem, counter, total, started):
    async with sem:
        target = row["image_path"]
        tasks = {
            "gpt52":  asyncio.create_task(call_openai(ENSEMBLE["gpt52"], target)),
            "gem3fp": asyncio.create_task(call_gemini(session, ENSEMBLE["gem3fp"], target)),
            "gpt41":  asyncio.create_task(call_openai(ENSEMBLE["gpt41"], target)),
        }
        preds = {}
        for key, t in tasks.items():
            raw, dt, err = await t
            parsed = parse(raw)
            preds[key] = {
                "model": ENSEMBLE[key],
                "raw_response": raw,
                "parsed": parsed,
                "predicted_score": parsed.get("score") if parsed else None,
                "predicted_category": parsed.get("category") if parsed else None,
                "elapsed_sec": round(dt, 2),
                "error": err,
            }
        rec = {
            "image_id": row["image_id"],
            "filename": row["filename"],
            "image_path": row["image_path"],
            "property_globalid": row["property_globalid"],
            "gt_score": int(row["gt_score"]) if row["gt_score"].isdigit() else None,
            "gt_category": row["gt_category"],
            "predictions": preds,
        }
        await write_record(rec)

        counter[0] += 1
        n = counter[0]
        if n % 20 == 0 or n == total:
            elapsed = time.time() - started
            rate = n / max(elapsed, 1e-9)
            eta = (total - n) / max(rate, 1e-9)
            s1 = preds["gpt52"]["predicted_score"] or "-"
            s2 = preds["gem3fp"]["predicted_score"] or "-"
            s3 = preds["gpt41"]["predicted_score"] or "-"
            print(f"  [{n:>4}/{total}]  last gt={rec['gt_score']} (gpt5.2={s1} gem3fp={s2} gpt4.1={s3})  "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f} min", flush=True)


async def main():
    with open(r"nogwh_manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Score 0 = unrated; keep all 1,280 (we'll exclude score-0 from accuracy later)
    done = load_done()
    pending = [r for r in rows if r["image_id"] not in done]
    print(f"Total images:  {len(rows)}")
    print(f"Already done:  {len(done)}")
    print(f"To run:        {len(pending)}")
    print(f"Models:        {', '.join(ENSEMBLE.values())}\n", flush=True)
    if not pending:
        print("Nothing to do."); return

    sem = asyncio.Semaphore(6)
    counter = [0]
    started = time.time()
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(process(r, session, sem, counter, len(pending), started) for r in pending))

    print(f"\nDone. Total time: {(time.time()-started)/60:.1f} min", flush=True)
    print(f"Output: {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
