"""How many consensus images MATCH ground truth — before vs after the new annotation.
Baseline P6 prompt, orig3 ensemble, 3-class consensus (all 3 agree on category)."""
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
def class3(s):
    if s in (1,2): return "H"
    if s==3: return "I"
    if s in (4,5): return "U"

# consensus images (all 3 agree 3-class)
cons=[]
for fn,r in base.items():
    if r["gt_score"] not in (1,2,3,4,5): continue
    preds=[r["predictions"][k]["predicted_score"] for k in MEM]
    if any(p is None for p in preds): continue
    cats=[class3(p) for p in preds]
    if len(set(cats))==1:
        cons.append((fn, cats[0], r["gt_score"]))

print(f"Images with 3-class consensus (all 3 agree): {len(cons)}\n")

def match_count(use_corrected):
    m=0
    for fn, aicat, city in cons:
        gt = rev.get(fn, city) if use_corrected else city
        if aicat==class3(gt): m+=1
    return m

before=match_count(False)
after=match_count(True)
n=len(cons)
print("="*58)
print("CONSENSUS IMAGES THAT MATCH GROUND TRUTH")
print("="*58)
print(f"  BEFORE (original City labels): {before}/{n} = {100*before/n:.1f}%")
print(f"  AFTER  (reviewer-corrected):   {after}/{n} = {100*after/n:.1f}%")
print(f"  Net additional matches gained: +{after-before}")
print(f"\n  (Of the {n} consensus images, {sum(1 for fn,_,_ in cons if fn in rev)} were in the audited set")
print(f"   and could be corrected; the rest kept their City label.)")

# breakdown of the gain by category
print(f"\n  By agreed category (matches before -> after):")
for c,lbl in [("H","Healthy"),("I","Intermediate"),("U","Unhealthy")]:
    grp=[(fn,ai,city) for fn,ai,city in cons if ai==c]
    b=sum(1 for fn,ai,city in grp if ai==class3(city))
    a=sum(1 for fn,ai,city in grp if ai==class3(rev.get(fn,city)))
    print(f"    {lbl:<13} {len(grp):>4} cases : {b} -> {a}  ({100*b/len(grp):.0f}% -> {100*a/len(grp):.0f}%)")
