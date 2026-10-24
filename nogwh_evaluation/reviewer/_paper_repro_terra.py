"""Three-run paper metrics for the deployed Terra/GPT-5.2/Gemini ensemble.

Uses the same reviewer-corrected labels and complete-case basis as
_paper_repro.py. Writes a machine-readable summary beside the Terra outputs.
"""
import collections
import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCHIVE = ROOT.parents[1] / "nogwh" / "archive_experiments"
OUT = ROOT / "terra_updated_three_run_summary.json"
RUNS = [
    ("run1", ROOT / "predictions_full.jsonl", ROOT / "predictions_full_terra.jsonl"),
    ("run2", ARCHIVE / "predictions_repro2.jsonl", ROOT / "predictions_full_terra_repro2.jsonl"),
    ("run3", ARCHIVE / "predictions_repro3.jsonl", ROOT / "predictions_full_terra_repro3.jsonl"),
]
MEMBERS = ("terra", "gpt52", "gem3fp")


def load_reviews(path):
    reviews = {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            score = (row.get("your_score") or "").strip()
            if score:
                reviews[row["filename"].strip()] = int(score)
    return reviews


REVIEWS = {
    **load_reviews(HERE / "reviews_f1_6.11.2026.csv"),
    **load_reviews(HERE / "reviews_f2a_6.18.2026.csv"),
}


def load_base(path):
    with path.open(encoding="utf-8") as handle:
        return {record["filename"]: record for record in map(json.loads, handle)}


def load_terra(path):
    with path.open(encoding="utf-8") as handle:
        records = map(json.loads, handle)
        return {
            record["filename"]: record["predicted_score"]
            for record in records
            if record.get("ok") and record.get("predicted_score") in (1, 2, 3, 4, 5)
        }


def class3(score):
    if score in (1, 2):
        return "Healthy"
    if score == 3:
        return "Intermediate"
    if score in (4, 5):
        return "Unhealthy"
    return None


def binary(score):
    if score in (1, 2):
        return "Healthy"
    if score in (4, 5):
        return "Unhealthy"
    return None


def majority(values):
    return collections.Counter(values).most_common(1)[0][0]


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0.0


def analyze(name, base_path, terra_path):
    base = load_base(base_path)
    terra = load_terra(terra_path)

    def truth(filename):
        return REVIEWS.get(filename, base[filename]["gt_score"])

    def score(filename, member):
        if member == "terra":
            return terra.get(filename)
        return base[filename]["predictions"][member]["predicted_score"]

    complete = [
        filename
        for filename, record in base.items()
        if record["gt_score"] in (1, 2, 3, 4, 5)
        and all(score(filename, member) in (1, 2, 3, 4, 5) for member in MEMBERS)
    ]
    model_metrics = {}
    for member in MEMBERS:
        model_metrics[member] = {
            "accuracy_3class_pct": pct(
                sum(class3(score(filename, member)) == class3(truth(filename)) for filename in complete),
                len(complete),
            ),
            "accuracy_5class_pct": pct(
                sum(score(filename, member) == truth(filename) for filename in complete),
                len(complete),
            ),
        }

    consensus = [
        filename
        for filename in complete
        if len({class3(score(filename, member)) for member in MEMBERS}) == 1
    ]
    disagreement = [filename for filename in complete if filename not in set(consensus)]
    exact_consensus = [
        filename
        for filename in complete
        if len({score(filename, member) for member in MEMBERS}) == 1
    ]

    consensus_correct = sum(
        class3(score(filename, "terra")) == class3(truth(filename))
        for filename in consensus
    )
    disagreement_correct = sum(
        majority([class3(score(filename, member)) for member in MEMBERS])
        == class3(truth(filename))
        for filename in disagreement
    )
    exact_correct = sum(
        score(filename, "terra") == truth(filename) for filename in exact_consensus
    )
    exact_disagreement = [filename for filename in complete if filename not in set(exact_consensus)]
    exact_disagreement_correct = sum(
        majority([score(filename, member) for member in MEMBERS]) == truth(filename)
        for filename in exact_disagreement
    )
    within_one = sum(
        abs(majority([score(filename, member) for member in MEMBERS]) - truth(filename)) <= 1
        for filename in consensus
    )
    majority_3class_correct = sum(
        majority([class3(score(filename, member)) for member in MEMBERS])
        == class3(truth(filename))
        for filename in complete
    )
    binary_base = [filename for filename in complete if binary(truth(filename)) is not None]
    binary_consensus = [
        filename
        for filename in binary_base
        if None not in [binary(score(filename, member)) for member in MEMBERS]
        and len({binary(score(filename, member)) for member in MEMBERS}) == 1
    ]
    binary_correct = sum(
        binary(score(filename, "terra")) == binary(truth(filename))
        for filename in binary_consensus
    )

    return {
        "run": name,
        "complete_n": len(complete),
        "models": model_metrics,
        "majority_3class_accuracy_pct": pct(majority_3class_correct, len(complete)),
        "consensus_n": len(consensus),
        "consensus_coverage_pct": pct(len(consensus), len(complete)),
        "consensus_accuracy_pct": pct(consensus_correct, len(consensus)),
        "disagreement_n": len(disagreement),
        "disagreement_accuracy_pct": pct(disagreement_correct, len(disagreement)),
        "exact_consensus_n": len(exact_consensus),
        "exact_consensus_coverage_pct": pct(len(exact_consensus), len(complete)),
        "exact_consensus_accuracy_pct": pct(exact_correct, len(exact_consensus)),
        "exact_disagreement_accuracy_pct": pct(
            exact_disagreement_correct, len(exact_disagreement)
        ),
        "within_one_on_3class_consensus_pct": pct(within_one, len(consensus)),
        "binary_base_n": len(binary_base),
        "binary_consensus_n": len(binary_consensus),
        "binary_consensus_accuracy_pct": pct(binary_correct, len(binary_consensus)),
        "consensus_group_counts": dict(
            collections.Counter(class3(score(filename, "terra")) for filename in consensus)
        ),
    }


def aggregate(runs):
    fields = (
        "majority_3class_accuracy_pct",
        "consensus_coverage_pct",
        "consensus_accuracy_pct",
        "disagreement_accuracy_pct",
        "exact_consensus_coverage_pct",
        "exact_consensus_accuracy_pct",
        "exact_disagreement_accuracy_pct",
        "within_one_on_3class_consensus_pct",
        "binary_consensus_accuracy_pct",
    )
    summary = {}
    for field in fields:
        values = [run[field] for run in runs]
        summary[field] = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "population_sd": statistics.pstdev(values),
        }
    return summary


def main():
    runs = [analyze(*run) for run in RUNS]
    result = {"ensemble": list(MEMBERS), "runs": runs, "aggregate": aggregate(runs)}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for run in runs:
        print(
            f"{run['run']}: n={run['complete_n']} "
            f"coverage={run['consensus_coverage_pct']:.1f}% "
            f"consensus_accuracy={run['consensus_accuracy_pct']:.1f}% "
            f"disagreement_accuracy={run['disagreement_accuracy_pct']:.1f}%"
        )
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
