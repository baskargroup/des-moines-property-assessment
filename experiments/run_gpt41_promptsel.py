"""
Run GPT-4.1 across all 7 prompt configs on the original 100-image benchmark,
with TRUE image few-shot: each "[IMAGE: <file> sent as base64]" marker in the
prompt file is replaced IN POSITION by the actual example image, and
"[TARGET IMAGE sent as base64]" by the target image. Zero-shot prompts (P1-P3)
have no markers, so the target image is appended at the end (matching the
original harness behavior of prompt-text + target image).

Model: gpt-4.1, temperature=0.0, max_tokens=8000 (same params as OpenAIVLM).
Resume-safe. Writes per-prompt results.csv + a combined summary
(5-class and 3-class accuracy). GPT-4.1 only (Qwen replacement).
"""
import os, csv, json, base64, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX_DIR = os.path.join(ROOT, "data", "download", "example_images")
load_dotenv(os.path.join(ROOT, ".env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1"

CONFIGS = [
    "config_01_zeroshot_original.json",
    "config_02_zeroshot_improved.json",
    "config_03_zeroshot_text_examples.json",
    "config_04_fewshot_1img.json",
    "config_05_fewshot_2img.json",
    "config_06_fewshot_5img.json",
    "config_07_fewshot_8img.json",
]
OUT_ROOT = os.path.join(ROOT, "data", "outputs", "gpt41_promptsel")
os.makedirs(OUT_ROOT, exist_ok=True)

MARKER = re.compile(r"\[IMAGE:\s*(?P<f>[^\]]+?)\s+sent as base64\]|\[TARGET IMAGE sent as base64\]")


def img_part(path):
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b}"}}


def build_content(prompt_text, target_path):
    parts, last, has_target = [], 0, False
    for m in MARKER.finditer(prompt_text):
        seg = prompt_text[last:m.start()]
        if seg.strip():
            parts.append({"type": "text", "text": seg})
        if m.group("f"):
            parts.append(img_part(os.path.join(EX_DIR, m.group("f").strip())))
        else:
            parts.append(img_part(target_path)); has_target = True
        last = m.end()
    tail = prompt_text[last:]
    if tail.strip():
        parts.append({"type": "text", "text": tail})
    if not has_target:
        parts.append(img_part(target_path))
    return parts


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


def call(content, retries=3):
    params = {"model": MODEL, "messages": [{"role": "user", "content": content}],
              "max_tokens": 8000, "temperature": 0.0}
    for a in range(retries):
        try:
            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content, None
        except Exception as e:
            if a < retries - 1:
                time.sleep(5 * (a + 1)); continue
            return None, str(e)[:150]
    return None, "retries_exhausted"


def c3(s):
    return "H" if s in (1, 2) else "I" if s == 3 else "U" if s in (4, 5) else None


def as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def run_config(cfg_name):
    cfg = json.load(open(os.path.join(ROOT, "experiments", cfg_name)))
    exp = cfg["experiment_name"]
    prompt = open(os.path.join(ROOT, cfg["prompt_file"]), encoding="utf-8").read()
    gt = load_gt(os.path.join(ROOT, cfg["ground_truth_file"]))
    images_dir = os.path.join(ROOT, cfg["images_dir"])
    n_img_markers = len([m for m in MARKER.finditer(prompt) if m.group("f")])
    out_dir = os.path.join(OUT_ROOT, exp)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")

    done = {}
    if os.path.exists(out_csv):
        for r in csv.DictReader(open(out_csv, encoding="utf-8")):
            done[r["filename"]] = r
    todo = [(fn, g) for fn, g in gt.items()
            if os.path.exists(os.path.join(images_dir, fn)) and fn not in done]
    print(f"[{exp}] example_images={n_img_markers}  total={len(gt)} done={len(done)} todo={len(todo)}", flush=True)

    results = list(done.values())

    def work(item):
        fn, g = item
        content = build_content(prompt, os.path.join(images_dir, fn))
        raw, err = call(content)
        return {"id": g["id"], "filename": fn, "ground_truth": g["ground_truth"],
                "predicted_score": parse_score(raw), "success": raw is not None, "error": err}

    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(work, it): it for it in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 20 == 0 or i == len(todo):
                    print(f"  [{exp}] {i}/{len(todo)}", flush=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "filename", "ground_truth", "predicted_score", "success", "error"])
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in w.fieldnames})

    n = len(results)
    parsed = sum(1 for r in results if as_int(r["predicted_score"]) is not None)
    e5 = sum(1 for r in results if as_int(r["predicted_score"]) == int(r["ground_truth"]))
    e3 = sum(1 for r in results if c3(as_int(r["predicted_score"])) == c3(int(r["ground_truth"])))
    return {"exp": exp, "n": n, "parsed": parsed,
            "acc5": round(100 * e5 / n, 1) if n else 0,
            "acc3": round(100 * e3 / n, 1) if n else 0}


def main():
    summary = [run_config(c) for c in CONFIGS]
    print("\n" + "=" * 60)
    print("GPT-4.1 PROMPT-SELECTION SUMMARY (original 100, image few-shot)")
    print("=" * 60)
    print(f"{'Prompt':<32}{'parsed':>8}{'5-class':>9}{'3-class':>9}")
    for s in summary:
        print(f"{s['exp']:<32}{s['parsed']:>8}{s['acc5']:>8.1f}%{s['acc3']:>8.1f}%")
    json.dump(summary, open(os.path.join(OUT_ROOT, "summary_gpt41.json"), "w"), indent=2)
    print(f"\nSaved -> {os.path.join(OUT_ROOT, 'summary_gpt41.json')}")


if __name__ == "__main__":
    main()
