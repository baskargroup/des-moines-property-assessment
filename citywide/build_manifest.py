"""Build a scoring manifest: one representative photo per parcel, joined to the
2022 baseline score and coordinates.

Sources (read-only):
  raw_data/gdb/PropertyCondition.gdb            PropertyCondition layer
      -> GlobalID (matches attachments), GUID (stable key across gdbs)
  raw_data/gdb/PropertyCondition_2026Copy.gdb  PropertyCondition layer
      -> GUID, PrimaryStrScore, x, y   (GlobalIDs were regenerated in this copy)
  raw_data/exports/PropertyCondition__ATTACH_Table.xlsx
      -> ATTACHMENTID, REL_GLOBALID, ATT_NAME  (photo -> original GlobalID)
  raw_data/extracted_images/                    the 81,758 extracted photos

Join chain: photo -> original GlobalID -> GUID -> 2026-copy score/x/y.

Usage:
  python build_manifest.py --out manifests/pilot_200.csv --stratified 40 --seed 42
  python build_manifest.py --out manifests/full.csv
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import fiona
import openpyxl

RAW = Path(os.environ.get("DSM_RAW_DATA", "raw_data"))
GDB_ORIGINAL = RAW / "gdb" / "PropertyCondition.gdb"
GDB_SCORED = RAW / "gdb" / "PropertyCondition_2026Copy.gdb"
ATTACH_XLSX = RAW / "exports" / "PropertyCondition__ATTACH_Table.xlsx"
IMAGES = RAW / "extracted_images"

FIELDS = ["image_id", "filename", "original_filename", "image_path",
          "property_globalid", "property_guid", "baseline_score", "x", "y",
          "photos_on_parcel"]


def load_parcels():
    """Return {original_GlobalID: {guid, baseline_score, x, y}}."""
    scored_by_guid = {}
    with fiona.open(GDB_SCORED, layer="PropertyCondition") as src:
        for rec in src:
            p = rec["properties"]
            scored_by_guid[p["GUID"]] = {
                "baseline_score": p.get("PrimaryStrScore"),
                "x": p.get("x"),
                "y": p.get("y"),
            }
    parcels = {}
    with fiona.open(GDB_ORIGINAL, layer="PropertyCondition") as src:
        for rec in src:
            p = rec["properties"]
            scored = scored_by_guid.get(p["GUID"], {})
            parcels[p["GlobalID"]] = {
                "guid": p["GUID"],
                "baseline_score": scored.get("baseline_score"),
                "x": scored.get("x"),
                "y": scored.get("y"),
            }
    return parcels


def load_attachments():
    """Return attachment rows with the exact filenames produced by extraction.

    extract_attachments.py iterated the attachment table and renamed repeated
    ATT_NAME values as name_1.ext, name_2.ext, etc. Reconstructing that sequence
    here recovers duplicate-named photos instead of dropping every occurrence.
    """
    ws = openpyxl.load_workbook(ATTACH_XLSX, read_only=True).active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    col = {name: i for i, name in enumerate(header)}
    by_parcel = defaultdict(list)
    name_counts = Counter()
    seen_names = Counter()
    for r in rows:
        att_name = r[col["ATT_NAME"]]
        if not att_name:
            continue
        occurrence = seen_names[att_name]
        seen_names[att_name] += 1
        if occurrence:
            base, ext = os.path.splitext(att_name)
            resolved_name = f"{base}_{occurrence}{ext}"
        else:
            resolved_name = att_name
        by_parcel[r[col["REL_GLOBALID"]]].append(
            (r[col["ATTACHMENTID"]], resolved_name, r[col["GLOBALID"]], att_name)
        )
        name_counts[att_name] += 1
    duplicate_names = {name for name, count in name_counts.items() if count > 1}
    return by_parcel, duplicate_names, sum(name_counts.values())


def build_rows(parcels, by_parcel):
    rows, skipped = [], Counter()
    on_disk = {p.name for p in IMAGES.iterdir()}  # one listing instead of 66K stat calls
    for gid, atts in by_parcel.items():
        if gid not in parcels:
            skipped["parcel_not_in_layer"] += 1
            continue
        # Representative photo: the earliest attachment for the parcel.
        atts = sorted(atts, key=lambda a: a[0])
        chosen = None
        for att_id, resolved_name, att_gid, original_name in atts:
            if resolved_name in on_disk:
                chosen = (resolved_name, att_gid, original_name)
                break
        if chosen is None:
            skipped["no_unambiguous_file"] += 1
            continue
        att_name, att_gid, original_name = chosen
        p = parcels[gid]
        rows.append({
            "image_id": att_gid,
            "filename": att_name,
            "original_filename": original_name,
            "image_path": str(IMAGES / att_name),
            "property_globalid": gid,
            "property_guid": p["guid"],
            "baseline_score": p["baseline_score"] if p["baseline_score"] is not None else "",
            "x": p["x"] if p["x"] is not None else "",
            "y": p["y"] if p["y"] is not None else "",
            "photos_on_parcel": len(atts),
        })
    return rows, skipped


def stratified_sample(rows, per_score, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        try:
            s = int(r["baseline_score"])
        except (TypeError, ValueError):
            continue
        if 1 <= s <= 5:
            buckets[s].append(r)
    sample = []
    for s in range(1, 6):
        pool = buckets.get(s, [])
        rng.shuffle(pool)
        sample.extend(pool[:per_score])
    rng.shuffle(sample)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stratified", type=int, default=0,
                    help="parcels per baseline score 1-5 (0 = all parcels)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parcels = load_parcels()
    by_parcel, duplicate_names, attachment_count = load_attachments()
    rows, skipped = build_rows(parcels, by_parcel)
    print(f"parcels in layer: {len(parcels):,}")
    print(f"parcels with attachments: {len(by_parcel):,}")
    print(f"attachments in export: {attachment_count:,}")
    print(f"duplicated original ATT_NAME values recovered: {len(duplicate_names):,}")
    print(f"usable parcels: {len(rows):,}   skipped: {dict(skipped)}")

    if args.stratified:
        rows = stratified_sample(rows, args.stratified, args.seed)
        print(f"stratified sample: {len(rows)} ({args.stratified} per score, seed {args.seed})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    dist = Counter(r["baseline_score"] for r in rows)
    print(f"wrote {len(rows):,} rows -> {out}   baseline distribution: {dict(sorted(dist.items(), key=str))}")


if __name__ == "__main__":
    main()
