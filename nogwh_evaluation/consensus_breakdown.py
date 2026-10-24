import json, collections

with open(r"predictions_full.jsonl", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]

rated = [r for r in rows if r["gt_score"] in (1,2,3,4,5)]
MODELS = ["gpt52", "gem3fp", "gpt41"]

def class3(s):
    if s in (1,2): return "Healthy"
    if s == 3: return "Intermediate"
    if s in (4,5): return "Unhealthy"
    return None

# Only consider images where ALL 3 models returned a score
complete = [r for r in rated if all(r["predictions"][m]["predicted_score"] is not None for m in MODELS)]
print(f"Total scored images: {len(rated)}")
print(f"Images where all 3 models responded: {len(complete)}")
print(f"(Gemini failed on {len(rated)-len(complete)} so those can't have 3-way consensus)\n")

# ---- 3-CLASS consensus levels ----
print("="*70)
print("3-CLASS CONSENSUS (Healthy / Intermediate / Unhealthy)")
print("="*70)
all_agree = two_agree = all_differ = 0
all_agree_correct = two_agree_correct = 0
for r in complete:
    cats = [class3(r["predictions"][m]["predicted_score"]) for m in MODELS]
    gt = class3(r["gt_score"])
    counts = collections.Counter(cats)
    top, topn = counts.most_common(1)[0]
    if topn == 3:
        all_agree += 1
        if top == gt: all_agree_correct += 1
    elif topn == 2:
        two_agree += 1
        if top == gt: two_agree_correct += 1
    else:
        all_differ += 1

n = len(complete)
print(f"\n  All 3 agree:        {all_agree:>4} / {n}  ({100*all_agree/n:.1f}%)   accuracy when agreed: {all_agree_correct}/{all_agree} = {100*all_agree_correct/max(all_agree,1):.1f}%")
print(f"  2 of 3 agree:       {two_agree:>4} / {n}  ({100*two_agree/n:.1f}%)   majority-correct:     {two_agree_correct}/{two_agree} = {100*two_agree_correct/max(two_agree,1):.1f}%")
print(f"  All 3 differ:       {all_differ:>4} / {n}  ({100*all_differ/n:.1f}%)   (no majority)")
print(f"\n  => Any consensus (>=2 agree): {all_agree+two_agree}/{n} = {100*(all_agree+two_agree)/n:.1f}%")

# ---- 5-CLASS consensus levels ----
print("\n" + "="*70)
print("5-CLASS CONSENSUS (exact score 1-5)")
print("="*70)
a5 = t5 = d5 = 0
a5c = t5c = 0
for r in complete:
    vals = [r["predictions"][m]["predicted_score"] for m in MODELS]
    gt = r["gt_score"]
    counts = collections.Counter(vals)
    top, topn = counts.most_common(1)[0]
    if topn == 3:
        a5 += 1
        if top == gt: a5c += 1
    elif topn == 2:
        t5 += 1
        if top == gt: t5c += 1
    else:
        d5 += 1
print(f"\n  All 3 agree (exact):  {a5:>4} / {n}  ({100*a5/n:.1f}%)   accuracy: {a5c}/{a5} = {100*a5c/max(a5,1):.1f}%")
print(f"  2 of 3 agree:         {t5:>4} / {n}  ({100*t5/n:.1f}%)   majority-correct: {t5c}/{t5} = {100*t5c/max(t5,1):.1f}%")
print(f"  All 3 differ:         {d5:>4} / {n}  ({100*d5/n:.1f}%)")

# ---- Consensus broken down by which 3-class category they agreed on ----
print("\n" + "="*70)
print("WHERE ALL 3 AGREE (3-class) — broken down by agreed category")
print("="*70)
by_cat = collections.defaultdict(lambda: [0,0])  # [count, correct]
for r in complete:
    cats = [class3(r["predictions"][m]["predicted_score"]) for m in MODELS]
    if len(set(cats)) == 1:
        c = cats[0]
        by_cat[c][0] += 1
        if c == class3(r["gt_score"]): by_cat[c][1] += 1
print(f"\n  {'Agreed category':<16} {'count':>8} {'correct':>10} {'precision':>12}")
for c in ["Healthy","Intermediate","Unhealthy"]:
    cnt, cor = by_cat[c]
    print(f"  {c:<16} {cnt:>8} {cor:>10} {100*cor/max(cnt,1):>11.1f}%")
