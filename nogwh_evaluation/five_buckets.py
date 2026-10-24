import json, collections

with open(r"predictions_full.jsonl", encoding="utf-8") as f:
    rows=[json.loads(l) for l in f]
rated=[r for r in rows if r["gt_score"] in (1,2,3,4,5)]
MODELS=["gpt52","gem3fp","gpt41"]

complete=[r for r in rated if all(r["predictions"][m]["predicted_score"] is not None for m in MODELS)]
incomplete=[r for r in rated if not all(r["predictions"][m]["predicted_score"] is not None for m in MODELS)]

# 5-class (exact score) agreement buckets
A=[]  # all 3 same exact AND == GT
B=[]  # all 3 same exact AND != GT
C1=[] # exactly 2 same exact, majority == GT
C2=[] # exactly 2 same exact, majority != GT
D=[]  # all 3 different
for r in complete:
    s=[r["predictions"][m]["predicted_score"] for m in MODELS]
    gt=r["gt_score"]
    cnt=collections.Counter(s)
    top,topn=cnt.most_common(1)[0]
    if topn==3:
        (A if top==gt else B).append(r)
    elif topn==2:
        (C1 if top==gt else C2).append(r)
    else:  # all different
        D.append(r)

print("="*64)
print("5-CLASS (EXACT SCORE) AGREEMENT BUCKETS")
print("="*64)
print(f"  Total images (all 3 responded): {len(complete)}")
print(f"  (plus {len(incomplete)} where a model failed to respond)\n")
print(f"  File 1  All 3 agree, but DIFFERENT from City:  {len(B):>4}")
print(f"  File 2  2 of 3 agree, but DIFFERENT from City:  {len(C2):>4}")
print(f"  File 3  All 3 models DIFFERENT from each other: {len(D):>4}")
print(f"  File 4  All 3 agree AND MATCH the City:         {len(A):>4}")
print(f"  File 5  2 of 3 agree AND MATCH the City:        {len(C1):>4}")
print(f"  {'':>40}{'-'*5}")
print(f"  {'sum':>40}{len(A)+len(B)+len(C1)+len(C2)+len(D):>5}")

# For File 1 (the most interesting), show GT distribution
print(f"\n  File 1 detail (all-3-agree-but-wrong) by what City said:")
for s in (1,2,3,4,5):
    n=sum(1 for r in B if r['gt_score']==s)
    print(f"    City={s}: {n}")
