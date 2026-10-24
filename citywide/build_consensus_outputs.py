"""Merge the three citywide model runs and split properties by the paper's consensus rule.

Consensus = GPT-5.6 Terra, GPT-5.2 and Gemini 3 Flash all put the photo in the same
group: Healthy (1-2), Intermediate (3) or Unhealthy (4-5).

Outputs (runs/citywide_consensus/):
    citywide_consensus.csv      all three models agree on the group
    citywide_no_consensus.csv   models disagree, or a model has no answer
    merged.jsonl.gz             one record per property with all three answers (for HTML/map builders)
    summary.json
"""
import csv
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "citywide_consensus"

MODELS = [
    ("terra", "GPT-5.6 Terra", RUNS / "citywide_terra" / "predictions.jsonl"),
    ("gpt52", "GPT-5.2", RUNS / "citywide_gpt52" / "predictions.jsonl"),
    ("gemini", "Gemini 3 Flash", RUNS / "citywide_gem3fp" / "predictions.jsonl"),
]
TERRA_FINAL = RUNS / "citywide_terra" / "stage_66701_results.csv"

GROUP = {1: "Healthy", 2: "Healthy", 3: "Intermediate", 4: "Unhealthy", 5: "Unhealthy"}
GROUP_LABEL = {"Healthy": "Healthy (1-2)", "Intermediate": "Intermediate (3)", "Unhealthy": "Unhealthy (4-5)"}
CATEGORY = {1: "Excellent", 2: "Good", 3: "Average", 4: "Starting Distress", 5: "Distressed"}


def as_score(value):
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def as_list(value):
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(v).strip() for v in items if str(v).strip() and str(v).strip().lower() not in ("none", "n/a")]


def compact_answer(record):
    parsed = record.get("parsed") or {}
    subscores = {}
    for key, value in (parsed.get("subscores") or {}).items():
        if isinstance(value, dict):
            subscores[key] = [value.get("score"), value.get("explanation", "")]
        else:
            subscores[key] = [value, ""]
    return {
        "score": as_score(record.get("predicted_score") or parsed.get("score")),
        "justification": parsed.get("justification") or parsed.get("reasoning") or "",
        "red_flags": as_list(parsed.get("red_flags")),
        "minor_issues": as_list(parsed.get("minor_issues")),
        "positive_signs": as_list(parsed.get("positive_signs")),
        "subscores": subscores,
    }


def load_model(path):
    """Last successful answer per image_id (runs contain retries and staged re-runs)."""
    answers, failures = {}, {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            image_id = record.get("image_id")
            if not image_id:
                continue
            if record.get("ok") and as_score(record.get("predicted_score") or (record.get("parsed") or {}).get("score")):
                answers[image_id] = compact_answer(record)
            else:
                failures[image_id] = str(record.get("error") or "no valid answer")[:300]
    return answers, failures


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    properties = {}
    with TERRA_FINAL.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            properties[row["image_id"]] = {
                "image_id": row["image_id"],
                "filename": row["filename"],
                "property_globalid": row["property_globalid"],
                "property_guid": row["property_guid"],
                "longitude": float(row["x"]) if row["x"] else None,
                "latitude": float(row["y"]) if row["y"] else None,
                "city_score": as_score(row["baseline_score"]),
            }
    print(f"properties: {len(properties):,}")

    answers, failures = {}, {}
    for key, name, path in MODELS:
        answers[key], failures[key] = load_model(path)
        covered = sum(1 for i in properties if i in answers[key])
        print(f"{name:<15} answers for {covered:,} of {len(properties):,}")

    columns_c = ["image_id", "filename", "property_globalid", "property_guid", "latitude", "longitude",
                 "city_score", "city_group", "agreed_group", "final_score", "final_category",
                 "all_three_same_score", "agrees_with_city_group"]
    columns_n = ["image_id", "filename", "property_globalid", "property_guid", "latitude", "longitude",
                 "city_score", "city_group", "status", "reason", "middle_score", "middle_category"]
    for key, name, _ in MODELS:
        columns_c += [f"{key}_score", f"{key}_justification", f"{key}_red_flags"]
        columns_n += [f"{key}_score", f"{key}_group", f"{key}_justification", f"{key}_red_flags"]

    stats = Counter()
    group_counts = Counter()
    with (OUT / "citywide_consensus.csv").open("w", encoding="utf-8-sig", newline="") as fc, \
            (OUT / "citywide_no_consensus.csv").open("w", encoding="utf-8-sig", newline="") as fn, \
            gzip.open(OUT / "merged.jsonl.gz", "wt", encoding="utf-8") as fm:
        writer_c = csv.DictWriter(fc, fieldnames=columns_c)
        writer_n = csv.DictWriter(fn, fieldnames=columns_n)
        writer_c.writeheader()
        writer_n.writeheader()

        for image_id, prop in properties.items():
            model_answers = {key: answers[key].get(image_id) for key, _, _ in MODELS}
            scores = {key: (a or {}).get("score") for key, a in model_answers.items()}
            available = [s for s in scores.values() if s is not None]
            middle = int(statistics.median_low(available)) if available else None
            missing = [name for key, name, _ in MODELS if scores[key] is None]
            groups = {GROUP[s] for s in available}
            city_group = GROUP.get(prop["city_score"], "")

            base = {
                "image_id": image_id, "filename": prop["filename"],
                "property_globalid": prop["property_globalid"], "property_guid": prop["property_guid"],
                "latitude": prop["latitude"], "longitude": prop["longitude"],
                "city_score": prop["city_score"] or "", "city_group": city_group,
            }
            if not missing and len(groups) == 1:
                status = "consensus"
                group = groups.pop()
                row = {
                    **base, "agreed_group": GROUP_LABEL[group], "final_score": middle,
                    "final_category": CATEGORY[middle],
                    "all_three_same_score": "yes" if len(set(available)) == 1 else "no",
                    "agrees_with_city_group": ("yes" if city_group == group else "no") if city_group else "",
                }
                for key, _, _ in MODELS:
                    a = model_answers[key]
                    row[f"{key}_score"] = a["score"]
                    row[f"{key}_justification"] = a["justification"]
                    row[f"{key}_red_flags"] = "; ".join(a["red_flags"])
                writer_c.writerow(row)
                group_counts[group] += 1
                if city_group:
                    stats["consensus_with_city"] += 1
                    stats["consensus_matches_city"] += city_group == group
            else:
                if missing:
                    status = "incomplete"
                    reason = "Incomplete - no answer from " + ", ".join(missing)
                else:
                    status = "disagree"
                    reason = "Models disagree - " + ", ".join(
                        f"{name} = {scores[key]} ({GROUP[scores[key]]})" for key, name, _ in MODELS)
                row = {**base, "status": status, "reason": reason,
                       "middle_score": middle or "", "middle_category": CATEGORY.get(middle, "")}
                for key, _, _ in MODELS:
                    a = model_answers[key] or {}
                    row[f"{key}_score"] = a.get("score") or ""
                    row[f"{key}_group"] = GROUP.get(a.get("score"), "")
                    row[f"{key}_justification"] = a.get("justification") or failures[key].get(image_id, "")
                    row[f"{key}_red_flags"] = "; ".join(a.get("red_flags") or [])
                writer_n.writerow(row)

            stats[status] += 1
            fm.write(json.dumps({**prop, "status": status, "middle": middle, "answers": model_answers},
                                separators=(",", ":")) + "\n")

    total = len(properties)
    summary = {
        "properties": total,
        "consensus": stats["consensus"],
        "disagree": stats["disagree"],
        "incomplete": stats["incomplete"],
        "consensus_pct": round(100 * stats["consensus"] / total, 1),
        "consensus_by_group": {GROUP_LABEL[g]: n for g, n in group_counts.items()},
        "consensus_matches_city_group_pct": round(
            100 * stats["consensus_matches_city"] / max(1, stats["consensus_with_city"]), 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for path in sorted(OUT.iterdir()):
        print(f"{path.name:<32} {path.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
