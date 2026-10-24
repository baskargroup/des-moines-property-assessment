"""Consensus accuracy on the 5-point scale allowing +-1 tolerance.
Baseline P6, orig3. Consensus = all 3 agree on EXACT score, OR on 3-class? 
Show both definitions, before/after corrected labels."""
import csv, json, collections

rev={}
with open(r"reviews_f1_6.11.2026.csv",encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        ys=row["your_score"].strip()
        if ys: rev[row["filename"].strip()]=int(ys)

def load(path):
    d={}
    with open(path,encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); d[r["filename"]]=r
    return d
base=load(r"../predictions_full.jsonl")
MEM=["gpt52","gem3fp","gpt41"]
def class3(s): return "H" if s in (1,2) else "I" if s==3 else "U"
def maj(v): return collections.Counter(v).most_common(1)[0][0]

# Definition 1: consensus = all 3 agree on 3-CLASS (623 cases). Use majority exact score as the prediction.
cons_cat=[]
# Definition 2: consensus = all 3 agree on EXACT score.
cons_exact=[]
for fn,r in base.items():
    if r["gt_score"] not in (1,2,3,4,5): continue
    preds=[r["predictions"][k]["predicted_score"] for k in MEM]
    if any(p is None for p in preds): continue
    if len(set(class3(p) for p in preds))==1:
        cons_cat.append((fn, maj(preds), r["gt_score"]))
    if len(set(preds))==1:
        cons_exact.append((fn, preds[0], r["gt_score"]))

def gt_of(fn, city, corrected): return rev.get(fn,city) if corrected else city

def report(name, cons):
    print(f"\n{name}  (n={len(cons)})")
    for corrected in (False,True):
        exact=adj=0
        for fn,pred,city in cons:
            gt=gt_of(fn,city,corrected)
            if pred==gt: exact+=1
            if abs(pred-gt)<=1: adj+=1
        tag="CORRECTED" if corrected else "ORIGINAL "
        print(f"  {tag}: exact 5-class {100*exact/len(cons):.1f}%   within +/-1 {100*adj/len(cons):.1f}%")

print("="*64)
print("CONSENSUS 5-CLASS ACCURACY — exact vs within +/-1")
print("="*64)
report("Consensus = all 3 agree on 3-CLASS (pred = majority score)", cons_cat)
report("Consensus = all 3 agree on EXACT score", cons_exact)
