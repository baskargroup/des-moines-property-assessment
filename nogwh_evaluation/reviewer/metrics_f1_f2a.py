"""Headline metrics under 3 label regimes:
  (A) Original City labels
  (B) Corrected with f1 only            (current presentation numbers)
  (C) Corrected with f1 + f2a           (new)
Baseline ensemble = predictions_full.jsonl (gpt52, gem3fp, gpt41)."""
import csv, json, collections

def load_rev(path):
    d={}
    with open(path,encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ys=(r.get("your_score") or "").strip()
            if ys: d[r["filename"].strip()]=int(ys)
    return d
f1=load_rev(r"reviews_f1_6.11.2026.csv")
f2=load_rev(r"reviews_f2a_6.18.2026.csv")
rev_f1=dict(f1)
rev_f12={**f1,**f2}
print(f"f1 corrections: {len(f1)}   f2a corrections: {len(f2)}   combined: {len(rev_f12)}\n")

base={}
with open(r"../predictions_full.jsonl",encoding="utf-8") as f:
    for line in f:
        r=json.loads(line); base[r["filename"]]=r
MEM=["gpt52","gem3fp","gpt41"]
def c3(s): return "H" if s in (1,2) else "I" if s==3 else "U" if s in (4,5) else None
def maj(v): return collections.Counter(v).most_common(1)[0][0]
def gb(s): return "good" if s in (1,2) else "bad" if s in (4,5) else None

rated=[fn for fn,r in base.items() if r["gt_score"] in (1,2,3,4,5)]
comp=[fn for fn in rated if all(base[fn]["predictions"][k]["predicted_score"] is not None for k in MEM)]

def gt(fn, rev): return rev.get(fn, base[fn]["gt_score"])

def metrics(rev):
    out={}
    # ensemble 3-class & 5-class
    c3cor=c5cor=0
    for fn in rated:
        g=gt(fn,rev)
        v=[base[fn]["predictions"][k]["predicted_score"] for k in MEM if base[fn]["predictions"][k]["predicted_score"] is not None]
        if not v: continue
        if maj([c3(x) for x in v])==c3(g): c3cor+=1
        if maj(v)==g: c5cor+=1
    out["ens3"]=100*c3cor/len(rated); out["ens5"]=100*c5cor/len(rated)
    # 3-class consensus
    cons=[fn for fn in comp if len({c3(base[fn]['predictions'][k]['predicted_score']) for k in MEM})==1]
    cc=sum(1 for fn in cons if c3(base[fn]["predictions"]["gpt52"]["predicted_score"])==c3(gt(fn,rev)))
    out["cons3"]=100*cc/len(cons); out["cons3_n"]=len(cons)
    # good vs bad majority (exclude Fair gt)
    gbset=[fn for fn in rated if gt(fn,rev) in (1,2,4,5)]
    gc=0
    for fn in gbset:
        v=[base[fn]["predictions"][k]["predicted_score"] for k in MEM if base[fn]["predictions"][k]["predicted_score"] is not None]
        if maj([gb(x) for x in v if gb(x)] or ["good"])==gb(gt(fn,rev)): gc+=1
    out["gb_maj"]=100*gc/len(gbset); out["gb_maj_n"]=len(gbset)
    # good vs bad consensus
    ag=cor=0
    for fn in comp:
        g=gt(fn,rev)
        if g not in (1,2,4,5): continue
        labs=[gb(base[fn]["predictions"][k]["predicted_score"]) for k in MEM]
        if None in labs: continue
        if len(set(labs))==1:
            ag+=1
            if labs[0]==gb(g): cor+=1
    out["gb_cons"]=100*cor/ag; out["gb_cons_n"]=ag
    return out

A=metrics({}); B=metrics(rev_f1); C=metrics(rev_f12)

rows=[("3-class accuracy (full, majority)","ens3"),
      ("5-class accuracy (full, majority)","ens5"),
      ("3-class consensus accuracy","cons3"),
      ("Good-vs-bad majority (Fair excl.)","gb_maj"),
      ("Good-vs-bad consensus","gb_cons")]
print(f"{'Metric':<38}{'Orig':>8}{'+f1':>9}{'+f1+f2a':>10}{'chg vs +f1':>11}")
print("-"*75)
for lbl,k in rows:
    print(f"{lbl:<38}{A[k]:>7.1f}%{B[k]:>8.1f}%{C[k]:>9.1f}%{C[k]-B[k]:>+9.1f}")
print()
print(f"consensus n (3-class): orig/f1/f12 all = {A['cons3_n']}")
print(f"good-vs-bad consensus n: orig={A['gb_cons_n']} f1={B['gb_cons_n']} f12={C['gb_cons_n']}")
print(f"good-vs-bad majority n:  orig={A['gb_maj_n']} f1={B['gb_maj_n']} f12={C['gb_maj_n']}")
