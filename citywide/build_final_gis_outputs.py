"""Build GIS-ready point files and multi-resolution hotspot grids."""
import csv
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs" / "citywide_terra"
SOURCE = RUN_DIR / "stage_66701_results.csv"
OUT_DIR = RUN_DIR / "delivery"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_FIELDS = [
    "image_id", "filename", "property_globalid", "property_guid", "longitude",
    "latitude", "baseline_score", "terra_score", "score_delta", "terra_category",
]
GRID_FIELDS = [
    "longitude", "latitude", "property_count", "terra_mean",
    "terra_poor_or_distressed_pct", "baseline_rated_count", "baseline_mean",
    "mean_score_delta", "terra_worse_pct", "terra_better_pct",
]
CONFIGS = [
    (0.01, 20, "coarse"),
    (0.005, 10, "medium"),
    (0.0025, 5, "fine"),
    (0.00125, 3, "ultra"),
]


def number(value, kind=float):
    if value in ("", None):
        return None
    return kind(value)


def load_rows():
    rows = []
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "longitude": number(raw["x"]),
                    "latitude": number(raw["y"]),
                    "baseline": number(raw["baseline_score"], int),
                    "terra": number(raw["terra_score"], int),
                    "delta": number(raw["score_delta"], int),
                }
            )
    return rows


def write_points_csv(rows):
    path = OUT_DIR / "terra_citywide_points.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "filename": row["filename"],
                    "property_globalid": row["property_globalid"],
                    "property_guid": row["property_guid"],
                    "longitude": row["longitude"],
                    "latitude": row["latitude"],
                    "baseline_score": row["baseline"],
                    "terra_score": row["terra"],
                    "score_delta": row["delta"],
                    "terra_category": row["terra_category"],
                }
            )
    return path


def write_geojson(rows):
    path = OUT_DIR / "terra_citywide_points.geojson"
    with path.open("w", encoding="utf-8") as handle:
        handle.write('{"type":"FeatureCollection","features":[')
        first = True
        for row in rows:
            if row["longitude"] is None or row["latitude"] is None:
                continue
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["longitude"], row["latitude"]],
                },
                "properties": {
                    "image_id": row["image_id"],
                    "property_globalid": row["property_globalid"],
                    "property_guid": row["property_guid"],
                    "filename": row["filename"],
                    "baseline_score": row["baseline"],
                    "terra_score": row["terra"],
                    "score_delta": row["delta"],
                    "terra_category": row["terra_category"],
                },
            }
            if not first:
                handle.write(",")
            handle.write(json.dumps(feature, separators=(",", ":")))
            first = False
        handle.write("]}")
    return path


def build_grid(rows, size, minimum):
    cells = defaultdict(list)
    for row in rows:
        lon, lat = row["longitude"], row["latitude"]
        if lon is None or lat is None:
            continue
        center_lon = round(round(lon / size) * size, 6)
        center_lat = round(round(lat / size) * size, 6)
        cells[(center_lon, center_lat)].append(row)

    output = []
    for (lon, lat), members in cells.items():
        if len(members) < minimum:
            continue
        terra_scores = [row["terra"] for row in members]
        rated = [row for row in members if row["baseline"] in (1, 2, 3, 4, 5)]
        deltas = [row["delta"] for row in rated]
        output.append(
            {
                "longitude": lon,
                "latitude": lat,
                "property_count": len(members),
                "terra_mean": round(sum(terra_scores) / len(terra_scores), 3),
                "terra_poor_or_distressed_pct": round(
                    100 * sum(score >= 4 for score in terra_scores) / len(terra_scores), 2
                ),
                "baseline_rated_count": len(rated),
                "baseline_mean": (
                    round(sum(row["baseline"] for row in rated) / len(rated), 3)
                    if rated
                    else None
                ),
                "mean_score_delta": (
                    round(sum(deltas) / len(deltas), 3) if deltas else None
                ),
                "terra_worse_pct": (
                    round(100 * sum(delta > 0 for delta in deltas) / len(deltas), 2)
                    if deltas
                    else None
                ),
                "terra_better_pct": (
                    round(100 * sum(delta < 0 for delta in deltas) / len(deltas), 2)
                    if deltas
                    else None
                ),
            }
        )
    return output


def write_grid_csv(path, cells, sort_key, reverse=True):
    ordered = sorted(
        cells,
        key=lambda cell: (
            cell[sort_key] is not None,
            cell[sort_key] if cell[sort_key] is not None else -999,
        ),
        reverse=reverse,
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRID_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


def main():
    rows = load_rows()
    point_csv = write_points_csv(rows)
    geojson = write_geojson(rows)

    all_grids = {}
    for size, minimum, name in CONFIGS:
        cells = build_grid(rows, size, minimum)
        all_grids[name] = {"size": size, "min_n": minimum, "cells": cells}
        print(f"{name}: {len(cells)} cells")

    grids_path = OUT_DIR / "terra_hotspots_multi.json"
    grids_path.write_text(
        json.dumps(all_grids, separators=(",", ":")), encoding="utf-8"
    )
    medium = all_grids["medium"]["cells"]
    write_grid_csv(
        OUT_DIR / "terra_condition_hotspots.csv",
        medium,
        "terra_mean",
    )
    write_grid_csv(
        OUT_DIR / "terra_change_hotspots.csv",
        medium,
        "mean_score_delta",
    )

    print(f"rows={len(rows)} with_coordinates={sum(r['longitude'] is not None for r in rows)}")
    for path in (
        point_csv,
        geojson,
        grids_path,
        OUT_DIR / "terra_condition_hotspots.csv",
        OUT_DIR / "terra_change_hotspots.csv",
    ):
        print(f"{path.name}: {path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
