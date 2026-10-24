"""
Analyze the reviewer's audit of File 1 (302 consensus-wrong cases):
  - How often does she side with the City vs the AI consensus vs neither?
  - Build corrected ground truth and recompute headline metrics.
"""
import csv, json, collections

REV=r"reviews_f1_6.11.2026.csv"
PRED=r"../predictions_full.jsonl"
CAL =r"../predictions_full_calib.jsonl"

def class3(s):
    if s in (1,2): return "H"
    if s==3: return "I"
    if s in (4,5): return "U"
LBL={"H":"Healthy","I":"Intermediate","U":"Unhealthy"}

# ---- load reviewer ----
rev={}
with open(REV,encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        fn=row["filename"].strip()
        ys=row["your_score"].strip()
        if not ys: continue
        rev[fn]={"score":int(ys),"city":int(row["city_score"]),
                 "comment":(row.get("your_comment") or "").strip()}
print(f"Reviewer rows with a score: {len(rev)}")

# ---- load predictions (baseline + calib), keyed by filename ----
def load(path):
    d={}
    with open(path,encoding="utf-8") as f:
        for line in f:
            r=json.loads(line); d[r["filename"]]=r
    return d
base=load(PRED); cal=load(CAL)
MEM=["gpt52","gem3fp","gpt41"]
def maj(v): return collections.Counter(v).most_common(1)[0][0]
def ens3(src,fn,mem=MEM):
    cats=[class3(src[fn]["predictions"][k]["predicted_score"]) for k in mem
          if src[fn]["predictions"][k]["predicted_score"] is not None]
    return maj(cats) if cats else None

# ---- 1) Who does the reviewer side with? (3-class) ----
side_city=side_ai=side_neither=0
side_city_exact=0
n_comment=0
for fn,rv in rev.items():
    if fn not in base: continue
    ai=ens3(base,fn)                      # AI consensus category (these are all-agree cases)
    city=class3(rv["city"])
    her=class3(rv["score"])
    if rv["comment"]: n_comment+=1
    if her==city and her!=ai: side_city+=1
    elif her==ai and her!=city: side_ai+=1
    elif her==city==ai: pass
    else: side_neither+=1
    if rv["score"]==rv["city"]: side_city_exact+=1

n=len([fn for fn in rev if fn in base])
print(f"\n{'='*62}\nWHO DOES THE REVIEWER SIDE WITH? (3-class, n={n})\n{'='*62}")
print(f"  Sides with CITY (against AI):     {side_city}  ({100*side_city/n:.0f}%)")
print(f"  Sides with AI (against City):     {side_ai}  ({100*side_ai/n:.0f}%)")
print(f"  Neither (her own 3rd category):   {side_neither}  ({100*side_neither/n:.0f}%)")
print(f"  Exact-score same as City:         {side_city_exact} ({100*side_city_exact/n:.0f}%)")
print(f"  Left a comment:                   {n_comment}")

# exact-score distribution of her changes
diff=collections.Counter()
for fn,rv in rev.items():
    diff[rv["score"]-rv["city"]]+=1
print(f"\n  Her score minus City score distribution:")
for d in sorted(diff): print(f"    {d:+d}: {diff[d]}")

# ---- 2) Corrected ground truth: replace GT with her score on these 302 ----
def corrected_gt(fn, default):
    return rev[fn]["score"] if fn in rev else default

def full_metrics(src,tag,mem=MEM):
    rated=[fn for fn,r in src.items() if r["gt_score"] in (1,2,3,4,5)]
    # original vs corrected
    for gt_mode in ("ORIGINAL","CORRECTED"):
        n3=n5=0; tot=0
        for fn in rated:
            gt=src[fn]["gt_score"] if gt_mode=="ORIGINAL" else corrected_gt(fn,src[fn]["gt_score"])
            e3=ens3(src,fn,mem)
            v=[src[fn]["predictions"][k]["predicted_score"] for k in mem
               if src[fn]["predictions"][k]["predicted_score"] is not None]
            e5=maj(v) if v else None
            tot+=1
            if e3==class3(gt): n3+=1
            if e5==gt: n5+=1
        print(f"  {tag:<18} {gt_mode:<10} 3-class: {100*n3/tot:.1f}%   5-class: {100*n5/tot:.1f}%")

print(f"\n{'='*62}\nHEADLINE METRICS WITH CORRECTED LABELS (her score replaces City\non the 302 audited; everything else unchanged)\n{'='*62}")
full_metrics(base,"BASELINE prompt")
full_metrics(cal,"CALIB prompt")

# consensus accuracy corrected (baseline, orig3, 3-class all-agree)
print(f"\n  Consensus (all-3-agree) accuracy, baseline prompt:")
for gt_mode in ("ORIGINAL","CORRECTED"):
    ag=agc=0
    by=collections.defaultdict(lambda:[0,0])
    for fn,r in base.items():
        if r["gt_score"] not in (1,2,3,4,5): continue
        preds=[r["predictions"][k]["predicted_score"] for k in MEM]
        if any(p is None for p in preds): continue
        cats=[class3(p) for p in preds]
        if len(set(cats))==1:
            ag+=1
            gt=r["gt_score"] if gt_mode=="ORIGINAL" else corrected_gt(fn,r["gt_score"])
            by[cats[0]][0]+=1
            if cats[0]==class3(gt): agc+=1; by[cats[0]][1]+=1
    line=f"    {gt_mode:<10} {agc}/{ag} = {100*agc/max(ag,1):.1f}%   by-cat:"
    for c in ("H","I","U"):
        cnt,cor=by[c]; line+=f"  {c}:{100*cor/max(cnt,1):.0f}%"
    print(line)
