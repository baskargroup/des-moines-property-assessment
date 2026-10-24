"""
Build a manifest CSV linking each extracted NOGWH image to:
  - its property GlobalID
  - the ground-truth PrimaryStrScore
  - the 3-class category (Healthy / Intermediate / Unhealthy)
  - the path to the extracted image
"""
import os, csv, glob, fiona
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
NOGWH_GDB = os.getenv(
    "NOGWH_GDB",
    str(WORKSPACE / "raw_data/gdb/PropertyCondition_Collected2025.gdb"),
)
IMG_DIR = os.getenv(
    "NOGWH_IMAGE_DIR",
    str(WORKSPACE / "raw_data/extracted_images_nogwh"),
)
OUT_CSV = str(Path(__file__).resolve().parent / "nogwh_manifest.csv")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# Load property scores
prop_score = {}
with fiona.open(NOGWH_GDB, layer="PropertyCondition") as src:
    for rec in src:
        p = rec["properties"]
        prop_score[p["GlobalID"]] = p["PrimaryStrScore"]

# Load attachments
images = []
with fiona.open(NOGWH_GDB, layer="PropertyCondition__ATTACH") as src:
    for rec in src:
        p = rec["properties"]
        rel = p["REL_GLOBALID"]
        score = prop_score.get(rel)
        att_name = p["ATT_NAME"] or "unknown.jpg"
        att_id = p["GLOBALID"]
        images.append({
            "att_globalid": att_id,
            "property_globalid": rel,
            "original_name": att_name,
            "score": score,
        })

# Match extracted files: extract_nogwh.py named them score{X}_{8charprop}_{base}{ext}
disk_files = os.listdir(IMG_DIR)
print(f"Disk files: {len(disk_files)}")
print(f"DB attachments: {len(images)}")

# Map: prop_8char -> original_name -> file_on_disk
def prop8(gid):
    return gid.strip("{}").replace("-", "")[:8]

disk_lookup = {}
for f in disk_files:
    if not f.startswith("score"):
        continue
    parts = f.split("_", 2)
    if len(parts) < 3:
        continue
    prop_part = parts[1]
    rest = parts[2]
    disk_lookup.setdefault((prop_part, rest), []).append(f)

# Also track files we've already used
used = set()
rows = []
unmatched = 0
for img in images:
    score_str = str(img["score"]) if img["score"] is not None else "NA"
    base, ext = os.path.splitext(img["original_name"])
    expected_rest = f"{base}{ext}"
    key = (prop8(img["property_globalid"]), expected_rest)
    candidates = disk_lookup.get(key, [])
    # pick first unused candidate
    chosen = None
    for c in candidates:
        if c not in used:
            chosen = c
            used.add(c)
            break
    if not chosen:
        # try with suffixed name (in case of dedup)
        unmatched += 1
        continue
    rows.append({
        "image_id": img["att_globalid"].strip("{}"),
        "filename": chosen,
        "image_path": os.path.join(IMG_DIR, chosen).replace("\\", "/"),
        "property_globalid": img["property_globalid"],
        "gt_score": score_str,
        "gt_category": (
            "Healthy" if score_str in ("1", "2") else
            "Intermediate" if score_str == "3" else
            "Unhealthy" if score_str in ("4", "5") else "Unrated"
        ),
        "original_name": img["original_name"],
    })

print(f"Matched: {len(rows)}, Unmatched: {unmatched}")

# Write CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Distribution
from collections import Counter
dist = Counter(r["gt_score"] for r in rows)
print("\nGround truth distribution in manifest:")
for s in sorted(dist):
    print(f"  Score {s}: {dist[s]:>5}")
print(f"  TOTAL    : {len(rows):>5}")
print(f"\nWrote: {OUT_CSV}")
