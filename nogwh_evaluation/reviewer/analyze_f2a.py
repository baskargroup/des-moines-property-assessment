"""Quick report on the new f2a reviewer file."""
import csv, re, collections, json

F2=r"reviews_f2a_6.18.2026.csv"
F1=r"reviews_f1_6.11.2026.csv"

def load(p):
    rows=[]
    with open(p,encoding="utf-8-sig") as f:
        for r in csv.DictReader(f): rows.append(r)
    return rows
f2=load(F2); f1=load(F1)
f1files={r["filename"].strip() for r in f1}

def parse_ai(s):
    return [int(x) for x in re.findall(r"(\d)", s or "")]

print(f"f2a rows: {len(f2)}")
filled=[r for r in f2 if r.get("your_score","").strip()]
print(f"rows with your_score filled: {len(filled)}")
overlap=sum(1 for r in f2 if r["filename"].strip() in f1files)
print(f"overlap with f1 (same filename): {overlap}")
print(f"NEW images (not in f1): {len(f2)-overlap}\n")

def c3(s): return "H" if s in (1,2) else "I" if s==3 else "U" if s in (4,5) else None

# Is this an agreement-control set or disagreement set?
agree_city=0; disagree_city=0; ai_consensus=0
chg=0; dir_counts=collections.Counter()
side_ai=0; side_city=0; side_neither=0
for r in filled:
    city=int(r["city_score"]); you=int(r["your_score"])
    ai=parse_ai(r["ai_scores"])
    if len(ai)>=3:
        cats=[c3(x) for x in ai[:3]]
        cons = len(set(cats))==1
        if cons: ai_consensus+=1
        # did AI (majority cat) agree with city?
        maj=collections.Counter(cats).most_common(1)[0][0]
        if maj==c3(city): agree_city+=1
        else: disagree_city+=1
        # who did reviewer side with (3-class)
        if c3(you)==maj and c3(you)!=c3(city): side_ai+=1
        elif c3(you)==c3(city) and c3(you)!=maj: side_city+=1
        elif c3(you)!=maj and c3(you)!=c3(city): side_neither+=1
    if you!=city:
        chg+=1; dir_counts[you-city]+=1

print("--- Composition (does AI majority agree with City?) ---")
print(f"  AI majority AGREES with City : {agree_city}")
print(f"  AI majority DISAGREES        : {disagree_city}")
print(f"  AI full consensus (all 3)    : {ai_consensus}")

print("\n--- Reviewer changes vs City ---")
print(f"  changed City score: {chg}/{len(filled)} ({100*chg/len(filled):.0f}%)")
print(f"  kept City score   : {len(filled)-chg}")
print("  direction (your - city):")
for d in sorted(dir_counts): print(f"    {d:+d}: {dir_counts[d]}")
healthier=sum(v for k,v in dir_counts.items() if k<0)
worse=sum(v for k,v in dir_counts.items() if k>0)
print(f"  -> healthier: {healthier}   worse: {worse}")

print("\n--- When reviewer changed, who was right (3-class) ---")
print(f"  sided with AI over City : {side_ai}")
print(f"  sided with City over AI : {side_city}")
print(f"  sided with neither      : {side_neither}")

# gt distribution by city score
print("\n--- City-score distribution of this set ---")
cd=collections.Counter(int(r["city_score"]) for r in filled)
for s in sorted(cd): print(f"  score {s}: {cd[s]}")
