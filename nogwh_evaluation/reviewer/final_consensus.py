"""Final consensus numbers (coverage + accuracy), original vs reviewer-corrected labels,
for both prompts (orig3 ensemble)."""
import csv, json, collections

def class3(s):
    if s in (1,2): return "H"
    if s==3: return "I"
    if s in (4,5): return "U"

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
for tag,path in [("BASELINE P6",r"../predictions_full.jsonl"),
                 ("CALIB",r"../predictions_full_calib.jsonl")]:
    src=load(path)
    comp=[fn for fn,r in src.items() if r["gt_score"] in (1,2,3,4,5)
          and all(r["predictions"][k]["predicted_score"] is not None for k in MEM)]
    print("="*66)
    print(f"{tag}  (n={len(comp)} images, all 3 models responded)")
    print("="*66)
    for gt_mode in ("ORIGINAL","CORRECTED"):
        ag=agc=0
        by=collections.defaultdict(lambda:[0,0])
        for fn in comp:
            preds=[src[fn]["predictions"][k]["predicted_score"] for k in MEM]
            cats=[class3(p) for p in preds]
            if len(set(cats))!=1: continue
            ag+=1
            gt=src[fn]["gt_score"] if gt_mode=="ORIGINAL" else rev.get(fn,src[fn]["gt_score"])
            by[cats[0]][0]+=1
            if cats[0]==class3(gt):
                agc+=1; by[cats[0]][1]+=1
        print(f"\n  {gt_mode} labels:")
        print(f"    Consensus happens:   {ag}/{len(comp)} = {100*ag/len(comp):.0f}% of images")
        print(f"    Accuracy when agree: {agc}/{ag} = {100*agc/max(ag,1):.1f}%")
        for c,l in [("H","Healthy"),("I","Intermediate"),("U","Unhealthy")]:
            cnt,cor=by[c]
            print(f"      agreed {l:<13}: {cnt:>4} cases  {100*cor/max(cnt,1):>5.1f}% correct")
    print()
