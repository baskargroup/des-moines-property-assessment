"""Build a CSV containing updated ground truth and all model predictions."""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "predictions_full.jsonl"
REVIEW_FILES = [
    ROOT / "reviewer" / "reviews_f1_6.11.2026.csv",
    ROOT / "reviewer" / "reviews_f2a_6.18.2026.csv",
]
OUTPUT = ROOT / "nogwh_updated_ground_truth_predictions.csv"
MODEL_KEYS = ("gpt52", "gem3fp", "gpt41")


def category(score):
    if score in (1, 2):
        return "Healthy"
    if score == 3:
        return "Intermediate"
    if score in (4, 5):
        return "Unhealthy"
    return "Unrated"


def load_updates():
    updates = {}
    for path in REVIEW_FILES:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                score = (row.get("your_score") or "").strip()
                if score:
                    updates[row["filename"]] = int(score)
    return updates


def ensemble_score(scores):
    """Combine available model scores without arbitrarily favoring a model."""
    available = [score for score in scores if score is not None]
    if not available:
        return "", "no_predictions"

    counts = Counter(available)
    score, count = counts.most_common(1)[0]
    if count >= 2:
        return score, "majority_vote"

    if len(available) == 3:
        return sorted(available)[1], "median_all_different"

    if len(available) == 1:
        return available[0], "single_available"

    return "", "two_models_disagree"


def main():
    updates = load_updates()
    with PREDICTIONS.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    fields = [
        "image_id",
        "filename",
        "property_globalid",
        "original_ground_truth",
        "original_ground_truth_category",
        "updated_ground_truth",
        "updated_ground_truth_category",
        "ground_truth_source",
        "ground_truth_changed",
        "gpt_5_2_predicted_score",
        "gemini_3_flash_predicted_score",
        "gpt_4_1_predicted_score",
        "available_prediction_count",
        "ensemble_predicted_score",
        "ensemble_method",
    ]

    output_rows = []
    for record in records:
        original = int(record["gt_score"])
        updated = updates.get(record["filename"], original)
        predictions = [
            record["predictions"][key].get("predicted_score") for key in MODEL_KEYS
        ]
        combined, method = ensemble_score(predictions)
        output_rows.append(
            {
                "image_id": record["image_id"],
                "filename": record["filename"],
                "property_globalid": record["property_globalid"],
                "original_ground_truth": original,
                "original_ground_truth_category": category(original),
                "updated_ground_truth": updated,
                "updated_ground_truth_category": category(updated),
                "ground_truth_source": (
                    "returned_score"
                    if record["filename"] in updates
                    else "original_score"
                ),
                "ground_truth_changed": (
                    "yes"
                    if record["filename"] in updates and updated != original
                    else "no"
                ),
                "gpt_5_2_predicted_score": (
                    predictions[0] if predictions[0] is not None else ""
                ),
                "gemini_3_flash_predicted_score": (
                    predictions[1] if predictions[1] is not None else ""
                ),
                "gpt_4_1_predicted_score": (
                    predictions[2] if predictions[2] is not None else ""
                ),
                "available_prediction_count": sum(
                    score is not None for score in predictions
                ),
                "ensemble_predicted_score": combined,
                "ensemble_method": method,
            }
        )

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT}")
    print(
        f"Returned scores: {sum(row['ground_truth_source'] == 'returned_score' for row in output_rows)}"
    )
    print(
        f"Changed scores: {sum(row['ground_truth_changed'] == 'yes' for row in output_rows)}"
    )
    print(
        f"Complete three-model predictions: "
        f"{sum(row['available_prediction_count'] == 3 for row in output_rows)}"
    )


if __name__ == "__main__":
    main()
