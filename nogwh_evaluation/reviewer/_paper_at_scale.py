"""Consolidated at-scale numbers for the paper (Results 2.3 + 2.4).
Regime: corrected labels (f1 + f2a). Run: predictions_full.jsonl (RUN 1).
Ensemble = gpt52, gem3fp, gpt41. Base = images with city gt in 1-5 AND all
three model predictions present (comp). Reports Wilson 95% CIs.
"""
import csv, json, collections, math

def load_rev(path):
    d = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ys = (r.get("your_score") or "").strip()
            if ys:
                d[r["filename"].strip()] = int(ys)
    return d

f1 = load_rev(r"reviews_f1_6.11.2026.csv")
f2 = load_rev(r"reviews_f2a_6.18.2026.csv")
REV = {**f1, **f2}

base = {}
with open(r"../predictions_full.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line); base[r["filename"]] = r

MEM = ["gpt52", "gem3fp", "gpt41"]
def c3(s): return "H" if s in (1, 2) else "I" if s == 3 else "U" if s in (4, 5) else None
def gb(s): return "good" if s in (1, 2) else "bad" if s in (4, 5) else None
def maj(v): return collections.Counter(v).most_common(1)[0][0]
def gt(fn): return REV.get(fn, base[fn]["gt_score"])

def wilson(k, n):
    if n == 0: return (0, 0, 0)
    p = k / n; z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (100 * p, 100 * (c - h), 100 * (c + h))

def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:5.1f}%  (95% CI {lo:.1f}-{hi:.1f}; {k}/{n})"

rated = [fn for fn, r in base.items() if r["gt_score"] in (1, 2, 3, 4, 5)]
comp = [fn for fn in rated if all(base[fn]["predictions"][k]["predicted_score"] is not None for k in MEM)]
print(f"rated (city gt 1-5): {len(rated)}    comp (all 3 preds present): {len(comp)}\n")

def preds(fn): return [base[fn]["predictions"][k]["predicted_score"] for k in MEM]

# ---------- Overall ensemble majority (over comp) ----------
m3 = sum(1 for fn in comp if maj([c3(p) for p in preds(fn)]) == c3(gt(fn)))
m5 = sum(1 for fn in comp if maj(preds(fn)) == gt(fn))
print("== ENSEMBLE MAJORITY (over comp = %d) ==" % len(comp))
print("  3-class majority:", fmt(m3, len(comp)))
print("  5-class majority:", fmt(m5, len(comp)))

# ---------- 3-class consensus (agree vs disagree) ----------
cons = [fn for fn in comp if len({c3(p) for p in preds(fn)}) == 1]
dis = [fn for fn in comp if fn not in cons]
cc = sum(1 for fn in cons if c3(preds(fn)[0]) == c3(gt(fn)))
dc = sum(1 for fn in dis if maj([c3(p) for p in preds(fn)]) == c3(gt(fn)))
print("\n== 3-CLASS CONSENSUS ==")
print(f"  coverage (agree): {100*len(cons)/len(comp):.1f}%  ({len(cons)}/{len(comp)})")
print("  acc | agree   :", fmt(cc, len(cons)))
print("  acc | disagree:", fmt(dc, len(dis)))

# within +-1 on the 3-class consensus subset (5-point tolerance, pred=majority score)
w1 = sum(1 for fn in cons if abs(maj(preds(fn)) - gt(fn)) <= 1)
ex = sum(1 for fn in cons if maj(preds(fn)) == gt(fn))
print("  5-pt exact | agree :", fmt(ex, len(cons)))
print("  within +-1 | agree :", fmt(w1, len(cons)))

# ---------- 5-class exact consensus ----------
cons5 = [fn for fn in comp if len(set(preds(fn))) == 1]
dis5 = [fn for fn in comp if fn not in cons5]
c5 = sum(1 for fn in cons5 if preds(fn)[0] == gt(fn))
d5 = sum(1 for fn in dis5 if maj(preds(fn)) == gt(fn))
print("\n== 5-CLASS EXACT CONSENSUS ==")
print(f"  coverage (agree): {100*len(cons5)/len(comp):.1f}%  ({len(cons5)}/{len(comp)})")
print("  acc | agree   :", fmt(c5, len(cons5)))
print("  acc | disagree:", fmt(d5, len(dis5)))

# ---------- Binary good-vs-bad consensus ----------
gbset = [fn for fn in comp if gt(fn) in (1, 2, 4, 5)]
bcons = [fn for fn in gbset if None not in [gb(p) for p in preds(fn)] and len({gb(p) for p in preds(fn)}) == 1]
bdis = [fn for fn in gbset if fn not in bcons]
bc = sum(1 for fn in bcons if gb(preds(fn)[0]) == gb(gt(fn)))
print("\n== BINARY (healthy vs unhealthy) CONSENSUS ==")
print(f"  good/bad-labeled base: {len(gbset)}")
print(f"  coverage (agree): {100*len(bcons)/len(gbset):.1f}%  ({len(bcons)}/{len(gbset)})")
print("  acc | agree   :", fmt(bc, len(bcons)))

# ---------- Per-class CONSENSUS accuracy (agree cases, by consensus label) ----------
print("\n== 3-CLASS CONSENSUS ACC BY CLASS (agree cases) ==")
for cls in ("H", "I", "U"):
    sub = [fn for fn in cons if c3(preds(fn)[0]) == cls]
    k = sum(1 for fn in sub if c3(gt(fn)) == cls)
    print(f"  consensus={cls}: ", fmt(k, len(sub)))

# ---------- Per-model per-class accuracy (over comp, by TRUE class) ----------
print("\n== PER-MODEL ACC BY TRUE 3-CLASS (over comp) ==")
for k in MEM:
    line = f"  {k:6s}: "
    for cls in ("H", "I", "U"):
        sub = [fn for fn in comp if c3(gt(fn)) == cls]
        corr = sum(1 for fn in sub if c3(base[fn]["predictions"][k]["predicted_score"]) == cls)
        p, lo, hi = wilson(corr, len(sub))
        line += f"{cls} {p:.1f}% (n={len(sub)})   "
    print(line)
