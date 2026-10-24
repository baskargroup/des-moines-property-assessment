"""Swap ensemble (GPT-5.2 + Gemini-3FP + Gemini-3.5) vs original (with GPT-4.1),
under corrected labels: 3-class consensus + good-vs-bad."""
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
g35 =load(r"../predictions_full_gemini35.jsonl")
cal =load(r"../predictions_full_calib.jsonl")

def class3(s):
    if s in (1,2): return "H"
    if s==3: return "I"
    if s in (4,5): return "U"
def gb(s):
    if s in (1,2): return "good"
    if s in (4,5): return "bad"
    return None
def maj(v): return collections.Counter(v).most_common(1)[0][0]

def get_baseline(fn,k):
    if k=="gem35":
        return g35[fn]["predicted_score"] if fn in g35 else None
    return base[fn]["predictions"][k]["predicted_score"]
def get_calib(fn,k):
    return cal[fn]["predictions"][k]["predicted_score"]

CONFIGS=[
  ("BASELINE / orig3 (4.1)",  get_baseline, ["gpt52","gem3fp","gpt41"], base),
  ("BASELINE / swap3 (3.5)",  get_baseline, ["gpt52","gem3fp","gem35"], base),
  ("CALIB / orig3 (4.1)",     get_calib,    ["gpt52","gem3fp","gpt41"], cal),
  ("CALIB / swap3 (3.5)",     get_calib,    ["gpt52","gem3fp","gem35"], cal),
]

print(f"{'Config':<26}{'labels':<11}{'3cls maj':>9}{'3cls cons':>11}{'cov':>6}{'GvB maj':>9}{'GvB cons':>10}{'cov':>6}")
print("-"*92)
for name,get,mem,src in CONFIGS:
    fns=[fn for fn,r in src.items() if r["gt_score"] in (1,2,3,4,5)
         and all(get(fn,k) is not None for k in mem)]
    for gt_mode in ("ORIGINAL","CORRECTED"):
        def gt_of(fn):
            g=src[fn]["gt_score"]
            if gt_mode=="CORRECTED" and fn in rev: g=rev[fn]
            return g
        # 3-class majority
        m3=100*sum(1 for fn in fns if maj([class3(get(fn,k)) for k in mem])==class3(gt_of(fn)))/len(fns)
        # 3-class consensus
        ag=agc=0
        for fn in fns:
            cats=[class3(get(fn,k)) for k in mem]
            if len(set(cats))==1:
                ag+=1
                if cats[0]==class3(gt_of(fn)): agc+=1
        c3=100*agc/max(ag,1); cov3=100*ag/len(fns)
        # good vs bad
        sub=[fn for fn in fns if gt_of(fn) in (1,2,4,5)]
        cor=inc=0
        for fn in sub:
            pg=gb(maj([get(fn,k) for k in mem])); gt=gb(gt_of(fn))
            if pg is None: continue
            if pg==gt: cor+=1
            else: inc+=1
        gvb=100*cor/max(cor+inc,1)
        ag2=agc2=0
        for fn in sub:
            labs=[gb(get(fn,k)) for k in mem]
            if None in labs: continue
            if len(set(labs))==1:
                ag2+=1
                if labs[0]==gb(gt_of(fn)): agc2+=1
        gvbc=100*agc2/max(ag2,1); cov2=100*ag2/max(len(sub),1)
        print(f"{name:<26}{gt_mode:<11}{m3:>8.1f}%{c3:>10.1f}%{cov3:>5.0f}%{gvb:>8.1f}%{gvbc:>9.1f}%{cov2:>5.0f}%")
    print()
