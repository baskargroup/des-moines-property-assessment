"""Paper-100 (our ensemble) metrics + comparison to paper's 93.3%, then aggregate with NOGWH-1280."""
import json, csv, collections

def load(p):
    d=[]
    for line in open(p,encoding="utf-8"): d.append(json.loads(line))
    return d
P=load("predictions_paper100.jsonl")
N=load("predictions_full.jsonl")

# corrected NOGWH labels
rev={}
for path in ["reviewer/reviews_f1_6.11.2026.csv","reviewer/reviews_f2a_6.18.2026.csv"]:
    for row in csv.DictReader(open(path,encoding="utf-8-sig")):
        if (row.get("your_score") or "").strip(): rev[row["filename"].strip()]=int(row["your_score"])

MEM=["gpt52","gem3fp","gpt41"]
def c3(s): return "H" if s in (1,2) else "I" if s==3 else "U" if s in (4,5) else None
def gb(s): return "good" if s in (1,2) else "bad" if s in (4,5) else None
def maj(v): return collections.Counter(v).most_common(1)[0][0]

def prep(recs, corrected=False):
    out=[]
    for r in recs:
        if r["gt_score"] not in (1,2,3,4,5): continue
        sc=[r["predictions"][k]["predicted_score"] for k in MEM]
        if any(x is None for x in sc): continue
        gt=rev.get(r["filename"], r["gt_score"]) if corrected else r["gt_score"]
        out.append((sc,gt))
    return out

def metrics(data, name):
    n=len(data)
    e3=sum(1 for sc,gt in data if maj([c3(x) for x in sc])==c3(gt))
    e5=sum(1 for sc,gt in data if maj(sc)==gt)
    cons=[(sc,gt) for sc,gt in data if len({c3(x) for x in sc})==1]
    c3c=sum(1 for sc,gt in cons if c3(sc[0])==c3(gt))
    cadj=sum(1 for sc,gt in cons if abs(maj(sc)-gt)<=1)
    # good vs bad consensus
    ag=cor=0
    for sc,gt in data:
        if gt not in (1,2,4,5): continue
        labs=[gb(x) for x in sc]
        if None in labs: continue
        if len(set(labs))==1:
            ag+=1
            if labs[0]==gb(gt): cor+=1
    print(f"\n{name}  (n={n}, consensus n={len(cons)})")
    print(f"  ensemble 3-class (100% cov) : {100*e3/n:5.1f}%")
    print(f"  ensemble 5-class            : {100*e5/n:5.1f}%")
    print(f"  CONSENSUS 3-class           : {100*c3c/len(cons):5.1f}%   (cov {100*len(cons)/n:.0f}%)")
    print(f"  CONSENSUS within +-1        : {100*cadj/len(cons):5.1f}%")
    print(f"  CONSENSUS good-vs-bad       : {100*cor/ag:5.1f}%   (n={ag})")
    return {"n":n,"e3":e3,"e5":e5,"cons":cons,"c3c":c3c,"cadj":cadj,"gb_ag":ag,"gb_cor":cor}

print("="*70); print("PAPER-100 with OUR ensemble (GPT-5.2 + Gemini-3-Flash + GPT-4.1)")
print("  paper reported 93.3% consensus 3-class (with Qwen as 3rd model)"); print("="*70)
pp=metrics(prep(P), "Paper-100 (paper GT)")

print("\n"+"="*70); print("NOGWH-1280"); print("="*70)
nn_o=metrics(prep(N,False), "NOGWH (original City labels)")
nn_c=metrics(prep(N,True),  "NOGWH (corrected f1+f2a labels)")

def aggregate(a,b,label):
    n=a["n"]+b["n"]
    e3=a["e3"]+b["e3"]; e5=a["e5"]+b["e5"]
    cons_n=len(a["cons"])+len(b["cons"]); c3c=a["c3c"]+b["c3c"]; cadj=a["cadj"]+b["cadj"]
    gb_ag=a["gb_ag"]+b["gb_ag"]; gb_cor=a["gb_cor"]+b["gb_cor"]
    print(f"\n{label}  (total n={n})")
    print(f"  ensemble 3-class (100% cov) : {100*e3/n:5.1f}%")
    print(f"  ensemble 5-class            : {100*e5/n:5.1f}%")
    print(f"  CONSENSUS 3-class           : {100*c3c/cons_n:5.1f}%   (n={cons_n}, cov {100*cons_n/n:.0f}%)")
    print(f"  CONSENSUS within +-1        : {100*cadj/cons_n:5.1f}%   (n={cons_n})")
    print(f"  CONSENSUS good-vs-bad       : {100*gb_cor/gb_ag:5.1f}%   (n={gb_ag})")

print("\n"+"="*70); print("AGGREGATE: PAPER-100 + NOGWH-1280  (FINAL COMBINED NUMBERS)"); print("="*70)
aggregate(pp,nn_o,"AGG with NOGWH ORIGINAL labels")
aggregate(pp,nn_c,"AGG with NOGWH CORRECTED labels")
