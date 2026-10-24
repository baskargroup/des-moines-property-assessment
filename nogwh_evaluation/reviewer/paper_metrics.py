"""All city-scale benchmark numbers reported in the paper, for every run.

Labels are the City scores with the expert-reviewed corrections applied. The
base is the set of rated images with a valid score from all three members.

Usage:
  python paper_metrics.py                 # corrected-prompt runs (runs/p6_corrected)
  python paper_metrics.py --original      # original paper runs, for comparison
"""
import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEMBERS = ("terra", "gpt52", "gem3fp")
ARCHIVE = ROOT.parents[1] / "nogwh" / "archive_experiments"

ORIGINAL_RUNS = {
    "run1": {"terra": ROOT / "predictions_full_terra.jsonl",
             "gpt52": ROOT / "predictions_full.jsonl", "gem3fp": ROOT / "predictions_full.jsonl"},
    "run2": {"terra": ROOT / "predictions_full_terra_repro2.jsonl",
             "gpt52": ARCHIVE / "predictions_repro2.jsonl", "gem3fp": ARCHIVE / "predictions_repro2.jsonl"},
    "run3": {"terra": ROOT / "predictions_full_terra_repro3.jsonl",
             "gpt52": ARCHIVE / "predictions_repro3.jsonl", "gem3fp": ARCHIVE / "predictions_repro3.jsonl"},
}


def corrected_runs(tag):
    d = ROOT / "runs" / tag
    return {run: {m: d / f"predictions_{m}_{run}.jsonl" for m in MEMBERS}
            for run in ("run1", "run2", "run3")}


def load_reviews():
    reviews = {}
    for name in ("reviews_f1_6.11.2026.csv", "reviews_f2a_6.18.2026.csv"):
        with (HERE / name).open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                s = (row.get("your_score") or "").strip()
                if s:
                    reviews[row["filename"].strip()] = int(s)
    return reviews


def load_scores(path, member):
    """filename -> score; reads both the combined and the per-member JSONL layouts."""
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "predictions" in rec:
                s = rec["predictions"][member]["predicted_score"]
            elif rec.get("ok"):
                s = rec["predicted_score"]
            else:
                continue
            if s in (1, 2, 3, 4, 5):
                out[rec["filename"]] = s
    return out


def c3(s):
    return "Healthy" if s in (1, 2) else "Intermediate" if s == 3 else "Unhealthy" if s in (4, 5) else None


def binary(s):
    return "Healthy" if s in (1, 2) else "Unhealthy" if s in (4, 5) else None


def strict_majority(values):
    value, count = collections.Counter(values).most_common(1)[0]
    return value if count >= 2 else None


def first_majority(values):
    return collections.Counter(values).most_common(1)[0][0]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def rate(k, n):
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None, "ci95": wilson(k, n)}


def analyze(files, truth_city, reviews):
    scores = {m: load_scores(files[m], m) for m in MEMBERS}
    truth = {f: reviews.get(f, g) for f, g in truth_city.items()}
    base = sorted(f for f in truth if all(f in scores[m] for m in MEMBERS))
    sc = lambda f: [scores[m][f] for m in MEMBERS]

    res = {"complete_n": len(base), "models": {}}
    for i, m in enumerate(MEMBERS):
        per_class = {}
        for cls in ("Healthy", "Intermediate", "Unhealthy"):
            sub = [f for f in base if c3(truth[f]) == cls]
            per_class[cls] = rate(sum(c3(scores[m][f]) == cls for f in sub), len(sub))
        res["models"][m] = {
            "acc3": rate(sum(c3(scores[m][f]) == c3(truth[f]) for f in base), len(base)),
            "acc5": rate(sum(scores[m][f] == truth[f] for f in base), len(base)),
            "within1": rate(sum(abs(scores[m][f] - truth[f]) <= 1 for f in base), len(base)),
            "per_true_class": per_class,
        }

    res["majority3_acc"] = rate(sum(first_majority([c3(s) for s in sc(f)]) == c3(truth[f]) for f in base), len(base))
    res["majority5_acc"] = rate(sum(first_majority(sc(f)) == truth[f] for f in base), len(base))

    cons = [f for f in base if len({c3(s) for s in sc(f)}) == 1]
    res["consensus3_coverage"] = rate(len(cons), len(base))
    res["consensus3_acc"] = rate(sum(c3(scores["terra"][f]) == c3(truth[f]) for f in cons), len(cons))
    maj = [f for f in base if f not in set(cons) and strict_majority([c3(s) for s in sc(f)])]
    res["disagree3_majority_acc"] = rate(
        sum(strict_majority([c3(s) for s in sc(f)]) == c3(truth[f]) for f in maj), len(maj))
    res["disagree3_no_majority_n"] = len(base) - len(cons) - len(maj)

    exact = [f for f in base if len(set(sc(f))) == 1]
    res["consensus5_coverage"] = rate(len(exact), len(base))
    res["consensus5_acc"] = rate(sum(scores["terra"][f] == truth[f] for f in exact), len(exact))
    maj5 = [f for f in base if f not in set(exact) and strict_majority(sc(f))]
    res["disagree5_majority_acc"] = rate(sum(strict_majority(sc(f)) == truth[f] for f in maj5), len(maj5))

    res["within1_on_consensus3"] = rate(sum(abs(first_majority(sc(f)) - truth[f]) <= 1 for f in cons), len(cons))

    bbase = [f for f in base if binary(truth[f])]
    bcons = [f for f in bbase if None not in [binary(s) for s in sc(f)] and len({binary(s) for s in sc(f)}) == 1]
    res["binary_base_n"] = len(bbase)
    res["binary_consensus_coverage"] = rate(len(bcons), len(bbase))
    res["binary_consensus_acc"] = rate(sum(binary(scores["terra"][f]) == binary(truth[f]) for f in bcons), len(bcons))
    hu_cons = [f for f in cons if c3(scores["terra"][f]) != "Intermediate"]
    res["unanimous_HU_n"] = len(hu_cons)
    res["unanimous_HU_on_intermediate_truth_n"] = sum(c3(truth[f]) == "Intermediate" for f in hu_cons)

    res["consensus_by_class"] = {}
    for cls in ("Healthy", "Intermediate", "Unhealthy"):
        sub = [f for f in cons if c3(scores["terra"][f]) == cls]
        res["consensus_by_class"][cls] = rate(sum(c3(truth[f]) == cls for f in sub), len(sub))
    res["true_class_n"] = dict(collections.Counter(c3(truth[f]) for f in base))

    trusted = cons + maj
    correct = (sum(c3(scores["terra"][f]) == c3(truth[f]) for f in cons)
               + sum(strict_majority([c3(s) for s in sc(f)]) == c3(truth[f]) for f in maj))
    res["ablation_2of3"] = {"coverage": rate(len(trusted), len(base)), "acc": rate(correct, len(trusted))}
    two = [f for f in base if c3(scores["gpt52"][f]) == c3(scores["gem3fp"][f])]
    res["ablation_gpt52_gem3fp"] = {
        "coverage": rate(len(two), len(base)),
        "acc": rate(sum(c3(scores["gpt52"][f]) == c3(truth[f]) for f in two), len(two)),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", action="store_true")
    ap.add_argument("--tag", default="p6_corrected")
    args = ap.parse_args()
    runs = ORIGINAL_RUNS if args.original else corrected_runs(args.tag)
    runs = {k: v for k, v in runs.items() if all(p.exists() for p in v.values())}

    with (ROOT / "nogwh_manifest.csv").open(encoding="utf-8-sig") as fh:
        truth_city = {r["filename"]: int(r["gt_score"]) for r in csv.DictReader(fh)
                      if r["gt_score"] in {"1", "2", "3", "4", "5"}}
    reviews = load_reviews()
    results = {run: analyze(files, truth_city, reviews) for run, files in runs.items()}

    keys = ["consensus3_coverage", "consensus3_acc", "majority3_acc", "majority5_acc",
            "binary_consensus_acc", "disagree3_majority_acc"]
    agg = {}
    for k in keys:
        vals = [r[k]["pct"] for r in results.values()]
        agg[k] = {"mean": round(statistics.mean(vals), 1), "min": min(vals), "max": max(vals),
                  "half_range": round((max(vals) - min(vals)) / 2, 1)}
    out = {"runs": results, "aggregate": agg}
    dest = ROOT / ("original_paper_metrics.json" if args.original else f"runs/{args.tag}/paper_metrics.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for run, r in results.items():
        m = r["models"]
        print(f"{run}: n={r['complete_n']}  acc3 terra/gpt52/gem={m['terra']['acc3']['pct']}/"
              f"{m['gpt52']['acc3']['pct']}/{m['gem3fp']['acc3']['pct']}  "
              f"cons3 {r['consensus3_coverage']['pct']}% ({r['consensus3_coverage']['k']}) acc "
              f"{r['consensus3_acc']['pct']}  maj-dis {r['disagree3_majority_acc']['k']}/"
              f"{r['disagree3_majority_acc']['n']}  binary {r['binary_consensus_acc']['k']}/"
              f"{r['binary_consensus_acc']['n']}  maj3 {r['majority3_acc']['pct']} maj5 {r['majority5_acc']['pct']}")
    print(json.dumps(agg, indent=1))
    print(f"Saved -> {dest}")


if __name__ == "__main__":
    main()
