"""Good (1,2) vs Bad (4,5) — GT=3 excluded, pred-3 set aside — with
original vs reviewer-corrected labels, baseline and calib prompts."""
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

MEM=["gpt52","gem3fp","gpt41"]
def gb(s):
    if s in (1,2): return "good"
    if s in (4,5): return "bad"
    return None
def maj(v): return collections.Counter(v).most_common(1)[0][0]

for tag,path in [("BASELINE P6",r"../predictions_full.jsonl"),
                 ("CALIB",r"../predictions_full_calib.jsonl")]:
    src=load(path)
    print("="*66); print(tag); print("="*66)
    for gt_mode in ("ORIGINAL","CORRECTED"):
        def gt_of(fn):
            g=src[fn]["gt_score"]
            if gt_mode=="CORRECTED" and fn in rev: g=rev[fn]
            return g
        sub=[fn for fn,r in src.items() if r["gt_score"] in (1,2,3,4,5) and gt_of(fn) in (1,2,4,5)]
        # majority
        cor=inc=aside=0
        for fn in sub:
            v=[src[fn]["predictions"][k]["predicted_score"] for k in MEM
               if src[fn]["predictions"][k]["predicted_score"] is not None]
            if not v: continue
            pg=gb(maj(v)); gt=gb(gt_of(fn))
            if pg is None: aside+=1
            elif pg==gt: cor+=1
            else: inc+=1
        # consensus
        ag=agc=0
        for fn in sub:
            preds=[src[fn]["predictions"][k]["predicted_score"] for k in MEM]
            if any(p is None for p in preds): continue
            labs=[gb(p) for p in preds]
            if None in labs: continue
            if len(set(labs))==1:
                ag+=1
                if labs[0]==gb(gt_of(fn)): agc+=1
        print(f"\n  {gt_mode} labels  (n={len(sub)} good-or-bad properties):")
        print(f"    Majority (pred-3 aside): {cor}/{cor+inc} = {100*cor/max(cor+inc,1):.1f}%  (aside={aside})")
        print(f"    Consensus:               {agc}/{ag} = {100*agc/max(ag,1):.1f}%  @ coverage {100*ag/len(sub):.0f}%")
    print()
