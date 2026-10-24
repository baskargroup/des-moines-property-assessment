"""3-run reproducibility on the SAME basis as the headline:
corrected labels (f1+f2a), base = all-three-present per run, over comp.
Runs: predictions_full (1), predictions_repro2 (2), predictions_repro3 (3).
Reports per-run 3-class majority, 5-class majority, and 3-class consensus."""
import csv, json, collections, statistics

def load_rev(path):
    d = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ys = (r.get("your_score") or "").strip()
            if ys:
                d[r["filename"].strip()] = int(ys)
    return d

REV = {**load_rev(r"reviews_f1_6.11.2026.csv"),
       **load_rev(r"reviews_f2a_6.18.2026.csv")}

MEM = ["gpt52", "gem3fp", "gpt41"]
def c3(s): return "H" if s in (1, 2) else "I" if s == 3 else "U" if s in (4, 5) else None
def maj(v): return collections.Counter(v).most_common(1)[0][0]

def load(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); d[r["filename"]] = r
    return d

RUNS = [("run1", r"../predictions_full.jsonl"),
        ("run2", r"../predictions_repro2.jsonl"),
        ("run3", r"../predictions_repro3.jsonl")]

maj3s, maj5s, cons3s, covs = [], [], [], []
print(f"{'Run':<6}{'comp':>6}{'maj3':>8}{'maj5':>8}{'cons_cov':>10}{'cons3':>8}")
for name, path in RUNS:
    base = load(path)
    def gt(fn): return REV.get(fn, base[fn]["gt_score"])
    def preds(fn): return [base[fn]["predictions"][k]["predicted_score"] for k in MEM]
    rated = [fn for fn, r in base.items() if r["gt_score"] in (1, 2, 3, 4, 5)]
    comp = [fn for fn in rated if all(base[fn]["predictions"][k]["predicted_score"] is not None for k in MEM)]
    m3 = 100 * sum(1 for fn in comp if maj([c3(p) for p in preds(fn)]) == c3(gt(fn))) / len(comp)
    m5 = 100 * sum(1 for fn in comp if maj(preds(fn)) == gt(fn)) / len(comp)
    cons = [fn for fn in comp if len({c3(p) for p in preds(fn)}) == 1]
    cov = 100 * len(cons) / len(comp)
    cons3 = 100 * sum(1 for fn in cons if c3(preds(fn)[0]) == c3(gt(fn))) / len(cons)
    maj3s.append(m3); maj5s.append(m5); cons3s.append(cons3); covs.append(cov)
    print(f"{name:<6}{len(comp):>6}{m3:>7.1f}%{m5:>7.1f}%{cov:>9.1f}%{cons3:>7.1f}%")

def band(xs):
    m = statistics.mean(xs)
    return m, (max(xs) - min(xs)) / 2, statistics.pstdev(xs)

for lbl, xs in [("3-class majority", maj3s), ("5-class majority", maj5s),
                ("consensus coverage", covs), ("3-class consensus", cons3s)]:
    m, halfrange, sd = band(xs)
    print(f"\n{lbl}: mean {m:.1f}%  half-range +/-{halfrange:.1f}  pstd {sd:.1f}  runs={[round(x,1) for x in xs]}")
