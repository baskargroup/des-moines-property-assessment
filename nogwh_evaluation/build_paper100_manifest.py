"""Build a run-manifest for the paper's 100-image set, matching the schema run_extra.py expects."""
import csv, os
GT=r"../data/datasets/ground_truth_100.csv"
IMGDIR=r"../data/download/test_100_images"
OUT=r"paper100_manifest.csv"

def cat(s):
    return {1:"Excellent",2:"Good",3:"Average",4:"Starting Distress",5:"Hazardous"}.get(s,"")

rows=[]; missing=[]
with open(GT,encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        fn=r["filename"].strip()
        p=os.path.join(IMGDIR,fn)
        if not os.path.exists(p): missing.append(fn); continue
        gt=int(r["ground_truth"])
        rows.append({
            "image_id": r["id"].strip(),
            "filename": fn,
            "image_path": p.replace("\\","/"),
            "property_globalid": r["id"].strip(),
            "gt_score": str(gt),
            "gt_category": cat(gt),
            "neighborhood": r.get("neighborhood",""),
            "source": r.get("source",""),
            "human_rater_score": r.get("human_rater_score",""),
        })
with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"manifest rows: {len(rows)}   missing images: {len(missing)}")
import collections
print("GT dist:", sorted(collections.Counter(int(r['gt_score']) for r in rows).items()))
print("neighborhoods:", sorted(collections.Counter(r['neighborhood'] for r in rows).items()))
if missing: print("MISSING:", missing[:10])
