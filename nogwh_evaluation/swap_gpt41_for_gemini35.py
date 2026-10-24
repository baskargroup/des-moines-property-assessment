import json, collections

# Main file: gpt52, gem3fp (gemini-3-flash-preview), gpt41
main={}
with open(r"predictions_full.jsonl", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line); main[r["image_id"]]=r
# gemini-3.5-flash file
g35={}
with open(r"predictions_full_gemini35.jsonl", encoding="utf-8") as f:
    for line in f:
        r=json.loads(line); g35[r["image_id"]]=r

ids=[i for i in main if i in g35 and main[i]["gt_score"] in (1,2,3,4,5)]
N=len(ids)

def class3(s):
    if s in (1,2): return "Healthy"
    if s==3: return "Intermediate"
    if s in (4,5): return "Unhealthy"
    return None

def get(i, who):
    if who=="gpt52":  return main[i]["predictions"]["gpt52"]["predicted_score"]
    if who=="gem3fp": return main[i]["predictions"]["gem3fp"]["predicted_score"]
    if who=="gpt41":  return main[i]["predictions"]["gpt41"]["predicted_score"]
    if who=="gem35":  return g35[i]["predicted_score"]

# Two ensembles to compare
ENS={
  "ORIGINAL  (gpt5.2 + gem-3-flash-prev + gpt4.1)":  ["gpt52","gem3fp","gpt41"],
  "SWAPPED   (gpt5.2 + gem-3-flash-prev + gem-3.5)": ["gpt52","gem3fp","gem35"],
}

def analyze(members):
    complete=[i for i in ids if all(get(i,m) is not None for m in members)]
    n=len(complete)
    # majority 3-class accuracy (over all ids, not just complete)
    maj_correct=0; maj_total=0
    for i in ids:
        cats=[class3(get(i,m)) for m in members if get(i,m) is not None]
        if not cats: continue
        maj_total+=1
        mv=collections.Counter(cats).most_common(1)[0][0]
        if mv==class3(main[i]["gt_score"]): maj_correct+=1
    # all-3 agree
    ag=agc=0
    by_cat=collections.defaultdict(lambda:[0,0])
    for i in complete:
        cats=[class3(get(i,m)) for m in members]
        if len(set(cats))==1:
            ag+=1
            by_cat[cats[0]][0]+=1
            if cats[0]==class3(main[i]["gt_score"]):
                agc+=1; by_cat[cats[0]][1]+=1
    return {
        "complete":n,
        "maj_acc":100*maj_correct/max(maj_total,1),
        "cov":100*ag/max(n,1), "agree":ag, "acc":100*agc/max(ag,1),
        "by_cat":by_cat,
    }

print("="*72)
print(f"SWAP TEST — does replacing GPT-4.1 with Gemini-3.5-Flash help? (n={N})")
print("="*72)

for name, members in ENS.items():
    a=analyze(members)
    print(f"\n  {name}")
    print(f"    Majority-vote 3-class accuracy:  {a['maj_acc']:.1f}%")
    print(f"    All-3-agree 3-class:  coverage {a['cov']:.0f}%  accuracy {a['acc']:.1f}%  ({a['agree']} cases)")
    print(f"    By agreed category:")
    for c in ["Healthy","Intermediate","Unhealthy"]:
        cnt,cor=a["by_cat"][c]
        print(f"        {c:<14} {cnt:>4} cases  {100*cor/max(cnt,1):>5.1f}% precision")

# Also pairwise within swapped ensemble
print("\n" + "="*72)
print("2-MODEL CONSENSUS options")
print("="*72)
complete_all=[i for i in ids if all(get(i,m) is not None for m in ["gpt52","gem3fp","gpt41","gem35"])]
def pair(a,b):
    ag=co=0
    for i in complete_all:
        if class3(get(i,a))==class3(get(i,b)):
            ag+=1
            if class3(get(i,a))==class3(main[i]["gt_score"]): co+=1
    return 100*ag/len(complete_all), 100*co/max(ag,1)
for a,b,lbl in [("gpt52","gem3fp","gpt5.2 + gemini-3-flash-preview"),
                ("gpt52","gem35","gpt5.2 + gemini-3.5-flash"),
                ("gem3fp","gem35","gemini-3-flash-preview + gemini-3.5-flash")]:
    cov,acc=pair(a,b)
    print(f"  {lbl:<46} coverage {cov:.0f}%  accuracy {acc:.1f}%")
