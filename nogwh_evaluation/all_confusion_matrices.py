import json, collections

with open(r"predictions_full.jsonl", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]
rated = [r for r in rows if r["gt_score"] in (1,2,3,4,5)]

MODELS = [("gpt52","gpt-5.2-2025-12-11"), ("gem3fp","gemini-3-flash-preview"), ("gpt41","gpt-4.1")]

def class3(s):
    if s in (1,2): return "Healthy"
    if s == 3: return "Intermediate"
    if s in (4,5): return "Unhealthy"
    return None

def maj5(scores):
    vals=[s for s in scores if s is not None]
    return collections.Counter(vals).most_common(1)[0][0] if vals else None
def maj3(scores):
    cats=[class3(s) for s in scores if s is not None]
    return collections.Counter(cats).most_common(1)[0][0] if cats else None


def print_5x5(title, getpred):
    print("\n" + "="*78)
    print(f"5-CLASS CONFUSION MATRIX — {title}")
    print("rows = ground truth, cols = predicted")
    print("="*78)
    cm = collections.defaultdict(collections.Counter)
    n_pred = 0
    for r in rated:
        p = getpred(r)
        if p is None: continue
        cm[r["gt_score"]][p] += 1
        n_pred += 1
    hdr = f"  {'GT/Pred':<9}"
    for c in (1,2,3,4,5): hdr += f"{('P'+str(c)):>8}"
    hdr += f"{'total':>8}{'recall':>9}"
    print(hdr)
    diag = 0
    for gt in (1,2,3,4,5):
        tot = sum(cm[gt][c] for c in (1,2,3,4,5))
        line = f"  GT={gt:<6}"
        for c in (1,2,3,4,5):
            line += f"{cm[gt][c]:>8}"
        rec = 100*cm[gt][gt]/max(tot,1)
        diag += cm[gt][gt]
        line += f"{tot:>8}{rec:>8.0f}%"
        print(line)
    # precision row
    prec_line = f"  {'prec':<9}"
    for c in (1,2,3,4,5):
        col = sum(cm[gt][c] for gt in (1,2,3,4,5))
        prec_line += f"{(str(round(100*cm[c][c]/max(col,1)))+'%'):>8}"
    print(prec_line)
    print(f"  Overall 5-class accuracy: {diag}/{n_pred} = {100*diag/max(n_pred,1):.1f}%")


def print_3x3(title, getpred3):
    print("\n" + "="*78)
    print(f"3-CLASS CONFUSION MATRIX — {title}")
    print("rows = ground truth, cols = predicted")
    print("="*78)
    classes = ["Healthy","Intermediate","Unhealthy"]
    cm = collections.defaultdict(collections.Counter)
    n_pred = 0
    for r in rated:
        p = getpred3(r)
        gt = class3(r["gt_score"])
        if p is None: continue
        cm[gt][p] += 1
        n_pred += 1
    hdr = f"  {'GT/Pred':<14}"
    for c in classes: hdr += f"{c:>14}"
    hdr += f"{'total':>8}{'recall':>9}"
    print(hdr)
    diag = 0
    for gt in classes:
        tot = sum(cm[gt][c] for c in classes)
        line = f"  {gt:<14}"
        for c in classes:
            line += f"{cm[gt][c]:>14}"
        rec = 100*cm[gt][gt]/max(tot,1)
        diag += cm[gt][gt]
        line += f"{tot:>8}{rec:>8.0f}%"
        print(line)
    prec_line = f"  {'precision':<14}"
    for c in classes:
        col = sum(cm[gt][c] for gt in classes)
        prec_line += f"{(str(round(100*cm[c][c]/max(col,1)))+'%'):>14}"
    print(prec_line)
    print(f"  Overall 3-class accuracy: {diag}/{n_pred} = {100*diag/max(n_pred,1):.1f}%")


# --- Per model ---
for key, label in MODELS:
    print_5x5(label, lambda r, k=key: r["predictions"][k]["predicted_score"])
    print_3x3(label, lambda r, k=key: class3(r["predictions"][k]["predicted_score"]))

# --- Ensemble majority vote ---
print_5x5("ENSEMBLE majority vote", lambda r: maj5([r["predictions"][k]["predicted_score"] for k,_ in MODELS]))
print_3x3("ENSEMBLE majority vote", lambda r: maj3([r["predictions"][k]["predicted_score"] for k,_ in MODELS]))
