import os, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = {
    "gpt-4.1": os.path.join(ROOT, "data/outputs/gpt41_promptsel"),
    "gpt-5.2": os.path.join(ROOT, "data/outputs/promptsel_images/gpt-5.2"),
    "gemini":  os.path.join(ROOT, "data/outputs/promptsel_images/gemini-3-flash"),
}
EXPS = ["01_zeroshot_original", "02_zeroshot_improved_class3", "03_zeroshot_text_examples",
        "04_fewshot_1img", "05_fewshot_2img", "06_fewshot_5img", "07_fewshot_8img"]


def as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def c3(s):
    return "H" if s in (1, 2) else "I" if s == 3 else "U" if s in (4, 5) else None


def load(d, exp):
    p = os.path.join(d, exp, "results.csv")
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        out[r["filename"]] = (as_int(r["predicted_score"]), as_int(r["ground_truth"]))
    return out


print(f"{'Prompt':<28}{'n':>4}{'cons_cov':>10}{'cons_acc':>10}{'maj3_acc':>10}{'5cls_cons_cov':>15}{'5cls_cons_acc':>15}")
for exp in EXPS:
    g4 = load(DIRS["gpt-4.1"], exp)
    g5 = load(DIRS["gpt-5.2"], exp)
    gm = load(DIRS["gemini"], exp)
    files = set(g4) & set(g5) & set(gm)
    n = 0
    cov3 = acc3 = 0
    maj_correct = 0
    cov5 = acc5 = 0
    for f in files:
        p4, gt = g4[f]
        p5, _ = g5[f]
        pm, _ = gm[f]
        if None in (p4, p5, pm, gt):
            continue
        n += 1
        gt3 = c3(gt)
        # 3-class consensus
        l4, l5, lm = c3(p4), c3(p5), c3(pm)
        if l4 == l5 == lm:
            cov3 += 1
            if l4 == gt3:
                acc3 += 1
        # 3-class majority
        labs = [l4, l5, lm]
        maj = max(set(labs), key=labs.count)
        if labs.count(maj) >= 2 and maj == gt3:
            maj_correct += 1
        # 5-class consensus (exact)
        if p4 == p5 == pm:
            cov5 += 1
            if p4 == gt:
                acc5 += 1
    print(f"{exp:<28}{n:>4}{100*cov3/n:>9.1f}%{(100*acc3/cov3 if cov3 else 0):>9.1f}%"
          f"{100*maj_correct/n:>9.1f}%{100*cov5/n:>14.1f}%{(100*acc5/cov5 if cov5 else 0):>14.1f}%")
