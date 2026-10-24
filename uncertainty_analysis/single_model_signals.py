"""Single-model uncertainty signals versus multi-model consensus on the NOGWH benchmark.

For each ensemble member, every signal ranks images from least to most uncertain.
Two summaries are reported per signal:
  - AUROC for detecting three-class errors (0.5 = uninformative)
  - three-class accuracy on the retained images when the same share of images is
    deferred as the three-model consensus rule defers (matched coverage)

Usage:
  python single_model_signals.py [--tag p6_corrected] [--run run1]
"""
import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "nogwh_evaluation"
sys.path.insert(0, str(EVAL / "reviewer"))
from paper_metrics import ORIGINAL_RUNS, load_reviews, c3, wilson  # noqa: E402

MEMBERS = ("terra", "gpt52", "gem3fp")
SUBSCORES = ("general_exterior_condition", "porch_entryway", "landscaping",
             "roof_gutters_downspouts", "windows", "extra_personal_touches")
NAMES = {"terra": "GPT-5.6 Terra", "gpt52": "GPT-5.2", "gem3fp": "Gemini-3-Flash"}


def load_member(path, member):
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "predictions" in rec:
                rec = {**rec["predictions"][member], "filename": rec["filename"]}
            elif not rec.get("ok"):
                continue
            if rec.get("predicted_score") not in (1, 2, 3, 4, 5):
                continue
            subs = (rec.get("parsed") or {}).get("subscores") or {}
            vals = [(subs.get(k) or {}).get("score") if isinstance(subs.get(k), dict) else subs.get(k)
                    for k in SUBSCORES]
            if all(v in (1, 2, 3, 4, 5) for v in vals):
                out[rec["filename"]] = (rec["predicted_score"], vals)
    return out


def entropy(vals):
    counts = collections.Counter(vals).values()
    n = len(vals)
    return -sum(c / n * math.log2(c / n) for c in counts)


def minmax(x):
    x = np.asarray(x, float)
    span = x.max() - x.min()
    return (x - x.min()) / span if span else np.zeros_like(x)


def signals(preds, subs, wrong):
    """Return {signal name: array where larger means more uncertain}."""
    subs = np.asarray(subs, float)
    preds = np.asarray(preds, float)
    mean, std = subs.mean(1), subs.std(1)
    rng = subs.max(1) - subs.min(1)
    gap = np.abs(preds - mean)
    boundary = -np.minimum(np.abs(mean - 2.5), np.abs(mean - 3.5))
    sig = {
        "Subscore variance": std,
        "Subscore range": rng,
        "Prediction-subscore gap": gap,
        "Boundary proximity": boundary,
        "Combined heuristic": (minmax(std) + minmax(rng) + minmax(boundary) + minmax(gap)) / 4,
        "Subscore entropy": np.array([entropy(list(v)) for v in subs]),
    }
    cls = np.array([c3(int(p)) for p in preds])
    proto = np.zeros(len(preds))
    for k in set(cls):
        m = cls == k
        proto[m] = np.linalg.norm(subs[m] - subs[m].mean(0), axis=1)
    sig["Prototype distance"] = proto
    feats = np.column_stack([subs, preds, mean, std, rng, gap, boundary])
    iso = IsolationForest(n_estimators=200, random_state=0).fit(feats)
    sig["Isolation forest"] = -iso.score_samples(feats)
    oof = np.zeros(len(preds))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(feats, wrong):
        rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(feats[tr], wrong[tr])
        oof[te] = rf.predict_proba(feats[te])[:, 1]
    sig["Random forest (5-fold CV)"] = oof
    return sig


def retained_accuracy(signal, correct, keep_n):
    order = np.argsort(signal, kind="stable")[:keep_n]
    k = int(correct[order].sum())
    return k, keep_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p6_corrected")
    ap.add_argument("--run", default="run1")
    ap.add_argument("--original", action="store_true", help="use the original paper runs")
    args = ap.parse_args()
    if args.original:
        files = ORIGINAL_RUNS[args.run]
        run_dir = EVAL
    else:
        run_dir = EVAL / "runs" / args.tag
        files = {m: run_dir / f"predictions_{m}_{args.run}.jsonl" for m in MEMBERS}
    members = [m for m in MEMBERS if files[m].exists()]
    data = {m: load_member(files[m], m) for m in members}

    with (EVAL / "nogwh_manifest.csv").open(encoding="utf-8-sig") as fh:
        city = {r["filename"]: int(r["gt_score"]) for r in csv.DictReader(fh)
                if r["gt_score"] in {"1", "2", "3", "4", "5"}}
    reviews = load_reviews()
    truth = {f: reviews.get(f, g) for f, g in city.items()}
    base = sorted(f for f in truth if all(f in data[m] for m in members))
    n = len(base)

    cons = [len({c3(data[m][f][0]) for m in members}) == 1 for f in base]
    keep_n = sum(cons)
    cons_correct = sum(c3(data[members[0]][f][0]) == c3(truth[f]) for f, c in zip(base, cons) if c)

    result = {"run": args.run, "members": members, "n": n,
              "consensus": {"coverage_pct": round(100 * keep_n / n, 1),
                            "accuracy_pct": round(100 * cons_correct / keep_n, 1),
                            "ci95": wilson(cons_correct, keep_n), "k": cons_correct, "n": keep_n},
              "models": {}}
    print(f"{args.run}: base n={n}, members={members}")
    print(f"Consensus: coverage {result['consensus']['coverage_pct']}% accuracy "
          f"{result['consensus']['accuracy_pct']}% {result['consensus']['ci95']}")
    for m in members:
        preds = [data[m][f][0] for f in base]
        subs = [data[m][f][1] for f in base]
        correct = np.array([c3(p) == c3(truth[f]) for p, f in zip(preds, base)])
        wrong = (~correct).astype(int)
        sig = signals(preds, subs, wrong)
        rows = {}
        for name, s in sig.items():
            k, kn = retained_accuracy(s, correct, keep_n)
            rows[name] = {"auroc": round(roc_auc_score(wrong, s), 3),
                          "matched_coverage_accuracy_pct": round(100 * k / kn, 1),
                          "ci95": wilson(k, kn)}
        result["models"][m] = {"overall_accuracy_pct": round(100 * correct.mean(), 1), "signals": rows}
        print(f"\n{NAMES[m]}  overall acc3={100 * correct.mean():.1f}%")
        for name, r in rows.items():
            print(f"  {name:<28} AUROC={r['auroc']:.3f}  acc@{keep_n}/{n}={r['matched_coverage_accuracy_pct']}% {r['ci95']}")

    prefix = "original_" if args.original else ""
    out = run_dir / f"{prefix}single_model_signals_{args.run}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
