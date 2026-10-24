"""Export successful campaign JSONL records to a reviewable flat CSV."""
import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def joined(values):
    return " | ".join(str(value) for value in (values or []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="citywide_terra")
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()

    run_dir = HERE / "runs" / args.run_id
    source = run_dir / "predictions.jsonl"
    output = run_dir / f"stage_{args.limit}_results.csv"

    latest = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("ok"):
                latest[record["image_id"]] = record

    records = sorted(
        (record for record in latest.values() if record["stage_limit"] <= args.limit),
        key=lambda record: record["timestamp"],
    )[: args.limit]

    fields = [
        "image_id", "filename", "property_globalid", "property_guid", "x", "y",
        "baseline_score", "terra_score", "score_delta", "terra_category",
        "justification", "red_flags", "minor_issues", "positive_signs",
        "cost_usd", "elapsed_sec", "prompt_tokens", "cached_tokens",
        "cache_write_tokens", "output_tokens", "reasoning_tokens",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            parsed = record["parsed"]
            usage = record["usage"]
            writer.writerow(
                {
                    "image_id": record["image_id"],
                    "filename": record["filename"],
                    "property_globalid": record["property_globalid"],
                    "property_guid": record["property_guid"],
                    "x": record["x"],
                    "y": record["y"],
                    "baseline_score": record["baseline_score"],
                    "terra_score": record["predicted_score"],
                    "score_delta": record["predicted_score"] - record["baseline_score"],
                    "terra_category": record["predicted_category"],
                    "justification": parsed.get("justification", ""),
                    "red_flags": joined(parsed.get("red_flags")),
                    "minor_issues": joined(parsed.get("minor_issues")),
                    "positive_signs": joined(parsed.get("positive_signs")),
                    "cost_usd": record["cost_usd"],
                    "elapsed_sec": record["elapsed_sec"],
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "cached_tokens": usage.get("cached_tokens", 0),
                    "cache_write_tokens": usage.get("cache_write_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                }
            )
    print(f"wrote {len(records)} records -> {output}")


if __name__ == "__main__":
    main()
