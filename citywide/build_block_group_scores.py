"""Mean condition score per 2020 Census block group, for each model, the three-model average and the City.

Needs build_consensus_outputs.py (merged.jsonl.gz) first.

Model-vs-City comparisons use only properties that have a City score, so both sides
describe the same houses. Block groups with fewer than MIN_PROPERTIES scored properties
(or fewer than MIN_PROPERTIES City-rated ones, for the City columns) are not reported.

Outputs (runs/citywide_block_groups/):
    block_group_scores.csv    one row per block group
    block_groups.geojson      WGS84 polygons with the scores (for the web map)
    comparison.json           block-group agreement of each model and the average with the City
"""
import csv
import gzip
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
SHAPEFILE = HERE.parent / "gis" / "census_2020_block_groups" / "CensusBlockGroupDSM2020.shp"
MERGED = HERE / "runs" / "citywide_consensus" / "merged.jsonl.gz"
OUT = HERE / "runs" / "citywide_block_groups"
MODELS = [("terra", "GPT-5.6 Terra"), ("gpt52", "GPT-5.2"), ("gemini", "Gemini 3 Flash")]
MIN_PROPERTIES = 5


def load_properties():
    rows = []
    with gzip.open(MERGED, "rt", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            scores = {key: (r["answers"].get(key) or {}).get("score") for key, _ in MODELS}
            rows.append({
                "image_id": r["image_id"], "latitude": r["latitude"], "longitude": r["longitude"],
                "city": r["city_score"], "consensus": r["status"] == "consensus", **scores,
            })
    frame = pd.DataFrame(rows)
    frame["average"] = frame[[key for key, _ in MODELS]].mean(axis=1)
    return frame


def agreement(x, y):
    diff = y - x
    return {
        "block_groups": int(len(x)),
        "pearson_r": round(float(x.corr(y)), 3),
        "spearman_rho": round(float(x.corr(y, method="spearman")), 3),
        "mean_difference": round(float(diff.mean()), 3),
        "mean_absolute_difference": round(float(diff.abs().mean()), 3),
        "rmse": round(math.sqrt(float((diff ** 2).mean())), 3),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zones = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
    zones = zones[["BlockGroup", "TRACTCE20", "BLKGRPCE20", "P0010001", "H0010001", "H0010003", "geometry"]]

    props = load_properties()
    located = props.dropna(subset=["latitude", "longitude"])
    points = gpd.GeoDataFrame(
        located, geometry=gpd.points_from_xy(located["longitude"], located["latitude"]), crs="EPSG:4326")
    joined = gpd.sjoin(points, zones[["BlockGroup", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    inside = joined.dropna(subset=["BlockGroup"])
    print(f"properties {len(props):,} · no coordinates {len(props) - len(located):,} · "
          f"outside every block group {len(joined) - len(inside):,} · assigned {len(inside):,}")

    keys = [key for key, _ in MODELS] + ["average"]
    rated = inside.dropna(subset=["city"])
    all_stats = inside.groupby("BlockGroup").agg(
        n=("image_id", "size"), consensus_pct=("consensus", "mean"), **{f"{k}_mean": (k, "mean") for k in keys})
    rated_stats = rated.groupby("BlockGroup").agg(
        n_city=("image_id", "size"), city_mean=("city", "mean"),
        **{f"{k}_rated_mean": (k, "mean") for k in keys})
    stats = all_stats.join(rated_stats, how="left")
    stats["n_city"] = stats["n_city"].fillna(0).astype(int)
    stats["consensus_pct"] *= 100

    shown = stats["n"] >= MIN_PROPERTIES
    stats.loc[~shown, [c for c in stats.columns if c.endswith("_mean") or c == "consensus_pct"]] = None
    city_ok = stats["n_city"] >= MIN_PROPERTIES
    city_cols = ["city_mean"] + [f"{k}_rated_mean" for k in keys]
    stats.loc[~city_ok, city_cols] = None
    stats["average_minus_city"] = stats["average_rated_mean"] - stats["city_mean"]

    comparable = stats.dropna(subset=["city_mean"])
    comparison = {
        "min_properties": MIN_PROPERTIES,
        "block_groups_total": int(len(zones)),
        "block_groups_shown": int(shown.sum()),
        "block_groups_compared_with_city": int(len(comparable)),
        "properties_assigned": int(len(inside)),
        "properties_compared_with_city": int(comparable["n_city"].sum()),
        "vs_city": {
            name: agreement(comparable["city_mean"], comparable[f"{key}_rated_mean"])
            for key, name in MODELS + [("average", "Average of 3 models")]
        },
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))

    table = zones.drop(columns="geometry").set_index("BlockGroup").join(stats, how="left")
    table["n"] = table["n"].fillna(0).astype(int)
    table["n_city"] = table["n_city"].fillna(0).astype(int)
    table = table.rename(columns={"P0010001": "population_2020", "H0010001": "housing_units_2020",
                                  "H0010003": "vacant_units_2020"})
    columns = ["TRACTCE20", "BLKGRPCE20", "population_2020", "housing_units_2020", "vacant_units_2020",
               "n", "consensus_pct", *[f"{k}_mean" for k in keys], "n_city", "city_mean",
               *[f"{k}_rated_mean" for k in keys], "average_minus_city"]
    table[columns].round(3).to_csv(OUT / "block_group_scores.csv", encoding="utf-8-sig",
                                   quoting=csv.QUOTE_MINIMAL)

    shapes = zones[["BlockGroup", "geometry"]].copy()
    shapes["geometry"] = shapes.geometry.simplify(0.00005, preserve_topology=True)
    shapes = shapes.merge(table[columns].round(3).reset_index(), on="BlockGroup", how="left")
    (OUT / "block_groups.geojson").write_text(shapes.to_json(drop_id=True, na="null"), encoding="utf-8")
    for path in sorted(OUT.iterdir()):
        print(f"{path.name:<26} {path.stat().st_size / 1e3:8.1f} KB")


if __name__ == "__main__":
    main()
