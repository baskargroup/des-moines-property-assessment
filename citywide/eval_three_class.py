"""Citywide coverage and 3-class agreement with the city baseline score.

Classes: 1-2 Healthy, 3 Intermediate, 4-5 Unhealthy. Baseline score 0 (unrated)
is excluded from accuracy. One prediction per image (first valid record).

Writes runs/three_class_eval.json.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {"terra": "citywide_terra", "gpt52": "citywide_gpt52", "gem3fp": "citywide_gem3fp"}
CLASSES = ["Healthy", "Intermediate", "Unhealthy"]


def class3(score) -> str | None:
    try:
        s = int(float(score))
    except (TypeError, ValueError):
        return None
    return {1: "Healthy", 2: "Healthy", 3: "Intermediate", 4: "Unhealthy", 5: "Unhealthy"}.get(s)


def load_predictions(run_id: str) -> dict[str, int]:
    preds: dict[str, int] = {}
    with (HERE / "runs" / run_id / "predictions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("ok")) != "True" or r["image_id"] in preds:
                continue
            if class3(r.get("predicted_score")) is not None:
                preds[r["image_id"]] = int(float(r["predicted_score"]))
    return preds


def metrics(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    correct = sum(t == p for t, p in pairs)
    cm = {t: Counter(p for tt, p in pairs if tt == t) for t in CLASSES}
    per_class = {}
    for c in CLASSES:
        tp = cm[c][c]
        support = sum(cm[c].values())
        predicted = sum(cm[t][c] for t in CLASSES)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"support": support, "predicted": predicted, "precision": precision,
                        "recall": recall, "f1": f1}
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "balanced_accuracy": sum(v["recall"] for v in per_class.values()) / len(CLASSES),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(CLASSES),
        "per_class": per_class,
        "confusion": {t: {p: cm[t][p] for p in CLASSES} for t in CLASSES},
    }


def main():
    with (HERE / "manifests" / "full.csv").open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    baseline = {r["image_id"]: class3(r["baseline_score"]) for r in manifest}
    rated = {k: v for k, v in baseline.items() if v}

    preds = {name: load_predictions(run) for name, run in RUNS.items()}
    majority = {}
    for img in baseline:
        votes = [class3(preds[m][img]) for m in RUNS if img in preds[m]]
        if len(votes) == 3:
            top, count = Counter(votes).most_common(1)[0]
            majority[img] = top if count >= 2 else "Intermediate"

    out = {"images": len(manifest), "baseline_rated": len(rated),
           "baseline_distribution": dict(Counter(rated.values())), "models": {}}
    for name in [*RUNS, "majority"]:
        cls = majority if name == "majority" else {k: class3(v) for k, v in preds[name].items()}
        pairs = [(rated[k], c) for k, c in cls.items() if k in rated]
        out["models"][name] = {
            "scored": len(cls),
            "coverage": len(cls) / len(manifest),
            "predicted_distribution": dict(Counter(cls.values())),
            **metrics(pairs),
        }
    (HERE / "runs" / "three_class_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"images {out['images']}  baseline-rated {out['baseline_rated']}  "
          f"baseline dist {out['baseline_distribution']}")
    for name, m in out["models"].items():
        pd = m["predicted_distribution"]
        print(f"\n{name}: scored {m['scored']} ({m['coverage']:.2%})  n_eval {m['n']}  "
              f"acc {m['accuracy']:.3f}  bal_acc {m['balanced_accuracy']:.3f}  "
              f"macroF1 {m['macro_f1']:.3f}")
        print("  predicted dist: " + "  ".join(f"{c} {pd.get(c, 0)} ({pd.get(c, 0) / m['scored']:.1%})"
                                               for c in CLASSES))
        for c in CLASSES:
            pc = m["per_class"][c]
            print(f"  {c:12s} support {pc['support']:6d}  recall {pc['recall']:.3f}  "
                  f"precision {pc['precision']:.3f}  f1 {pc['f1']:.3f}  "
                  f"confusion {m['confusion'][c]}")


if __name__ == "__main__":
    main()
