"""Full accuracy report for the 1,280-image NOGWH run (Ensemble B)."""
import json, collections

PATH = r"predictions_full.jsonl"
with open(PATH, encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]

# Exclude score-0 (unrated) from accuracy
rated = [r for r in rows if r["gt_score"] in (1, 2, 3, 4, 5)]
print(f"Total records: {len(rows)}   Scored (1-5) for eval: {len(rated)}\n")

MODELS = ["gpt52", "gem3fp", "gpt41"]
LABELS = {"gpt52": "gpt-5.2-2025-12-11", "gem3fp": "gemini-3-flash-preview", "gpt41": "gpt-4.1"}

def class3(s):
    if s in (1, 2): return "Healthy"
    if s == 3:      return "Intermediate"
    if s in (4, 5): return "Unhealthy"
    return None


print("="*84)
print("1. PER-MODEL ACCURACY  (n=%d scored images)" % len(rated))
print("="*84)
print(f"\n{'Model':<28} {'Success':<11} {'5-class':<16} {'3-class':<16} {'avg s':<7}")
print("-"*84)
for m in MODELS:
    succ = [r for r in rated if r["predictions"][m]["predicted_score"] is not None]
    n5 = sum(1 for r in succ if r["predictions"][m]["predicted_score"] == r["gt_score"])
    n3 = sum(1 for r in succ if class3(r["predictions"][m]["predicted_score"]) == class3(r["gt_score"]))
    avg = sum(r["predictions"][m]["elapsed_sec"] for r in rated) / len(rated)
    print(f"{LABELS[m]:<28} {len(succ)}/{len(rated):<7} "
          f"{n5}/{len(succ)}={100*n5/max(len(succ),1):.1f}%{'':<3} "
          f"{n3}/{len(succ)}={100*n3/max(len(succ),1):.1f}%{'':<3} {avg:.1f}")


print("\n" + "="*84)
print("2. 5-CLASS PER-SCORE ACCURACY")
print("="*84)
print(f"\n{'Model':<28}  GT=1     GT=2     GT=3     GT=4     GT=5      TOTAL")
print("-"*84)
for m in MODELS:
    line = f"{LABELS[m]:<28}"
    ct = st = 0
    for s in (1,2,3,4,5):
        g = [r for r in rated if r["gt_score"] == s]
        c = sum(1 for r in g if r["predictions"][m]["predicted_score"] == s)
        st += len(g); ct += c
        line += f"  {c}/{len(g)}={100*c/max(len(g),1):.0f}%"
    line += f"   {ct}/{st}={100*ct/max(st,1):.0f}%"
    print(line)


print("\n" + "="*84)
print("3. 3-CLASS CONFUSION MATRICES  (row=GT, col=pred)")
print("="*84)
for m in MODELS:
    print(f"\n--- {LABELS[m]} ---")
    cm = collections.defaultdict(collections.Counter)
    for r in rated:
        gt = class3(r["gt_score"]); pr = class3(r["predictions"][m]["predicted_score"])
        if gt and pr: cm[gt][pr] += 1
    classes = ["Healthy","Intermediate","Unhealthy"]
    lbl = "GT \\ Pred"
    hdr = f"  {lbl:<14}"
    for c in classes: hdr += f"{c:>14}"
    hdr += f"{'recall':>10}"
    print(hdr)
    for gt in classes:
        tot = sum(cm[gt][c] for c in classes)
        line = f"  {gt:<14}"
        for c in classes:
            n = cm[gt][c]
            line += f"{f'{n} ({100*n/max(tot,1):.0f}%)':>14}"
        diag = cm[gt][gt]
        line += f"{f'{100*diag/max(tot,1):.0f}%':>10}"
        print(line)


print("\n" + "="*84)
print("4. ENSEMBLE: 3-MODEL MAJORITY VOTE + CONSENSUS SIGNAL")
print("="*84)

def maj5(scores):
    vals = [s for s in scores if s is not None]
    return collections.Counter(vals).most_common(1)[0][0] if vals else None

def maj3(scores):
    cats = [class3(s) for s in scores if s is not None]
    return collections.Counter(cats).most_common(1)[0][0] if cats else None

n5 = n3 = succ = 0
ag3 = ag3c = ag5 = ag5c = 0
for r in rated:
    scores = [r["predictions"][m]["predicted_score"] for m in MODELS]
    vals = [s for s in scores if s is not None]
    if not vals: continue
    succ += 1
    if maj5(scores) == r["gt_score"]: n5 += 1
    if maj3(scores) == class3(r["gt_score"]): n3 += 1
    cats = [class3(s) for s in scores]
    if len(vals) == 3 and len(set(cats)) == 1:
        ag3 += 1
        if cats[0] == class3(r["gt_score"]): ag3c += 1
    if len(vals) == 3 and len(set(vals)) == 1:
        ag5 += 1
        if vals[0] == r["gt_score"]: ag5c += 1

print(f"\n  Scored images: {succ}")
print(f"\n  Majority vote:")
print(f"    5-class:  {n5}/{succ} = {100*n5/succ:.1f}%")
print(f"    3-class:  {n3}/{succ} = {100*n3/succ:.1f}%")
print(f"\n  Consensus signal (all 3 models agree):")
print(f"    3-class:  coverage {ag3}/{succ}={100*ag3/succ:.0f}%   accuracy {ag3c}/{ag3}={100*ag3c/max(ag3,1):.1f}%")
print(f"    5-class:  coverage {ag5}/{succ}={100*ag5/succ:.0f}%   accuracy {ag5c}/{ag5}={100*ag5c/max(ag5,1):.1f}%")

print("\n" + "="*84)
print("5. EMPTY / FAILED RESPONSES")
print("="*84)
for m in MODELS:
    fails = [r for r in rows if r["predictions"][m]["predicted_score"] is None]
    print(f"  {LABELS[m]:<28} {len(fails)} failures")

print("\n" + "="*84)
print("COMPARISON TO PAPER (paper benchmark = 100 stratified DSM images)")
print("="*84)
print("                              PAPER      OUR NOGWH (1,280)")
print(f"  3-class all-3-agree acc      93.3%      {100*ag3c/max(ag3,1):.1f}%")
print(f"  3-class agreement coverage   60.0%      {100*ag3/succ:.0f}%")
print("  (single-model 3-class best   84.0%      see section 1)")
