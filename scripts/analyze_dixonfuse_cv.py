"""Three-way cross-validated DixonFuse comparison (reviewer item 4).

At each budget N in {2,4,8}, collects per-subject Dice (averaged over repeats -> one
value per subject, n=25) for the four-channel baseline (dixon4), the fat-fraction input
without a gate (dixon5), the fat-fraction input with the squeeze-and-excitation gate
(dixonfuse), and, when available, the gate without fat-fraction (gateonly). Reports
subject-level means, subject-clustered 95% bootstrap CIs, and Holm-corrected paired
comparisons: baseline vs fat-fraction, baseline vs fat-fraction+gate, and
fat-fraction vs fat-fraction+gate. Writes results/dixonfuse_cv.json.
"""
import os, glob, json
from collections import defaultdict
import numpy as np
from scipy import stats
import config as C
R = C.RESULTS


def collect(tag, n):
    per = defaultdict(list)
    for f in glob.glob(os.path.join(R, f"cv_thigh_{tag}_n{n}_f*_r*.json")):
        for s, v in json.load(open(f))["per_subject_dice"].items():
            per[s].append(v)
    return {s: float(np.mean(v)) for s, v in per.items()}


def boot_ci(vals, n=5000):
    x = np.asarray(vals, float)
    idx = np.random.RandomState(0).randint(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); prev = 0
    for rank, i in enumerate(order):
        adj[i] = max(prev, min(1.0, (m - rank) * ps[i])); prev = adj[i]
    return adj


def paired(a, b):
    keys = sorted(set(a) & set(b))
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def main():
    tags = {"dixon4": "method-dixon4", "dixon5": "method-dixon5",
            "dixonfuse": "method-dixonfuse", "gateonly": "method-gateonly"}
    out = {}
    for N in [2, 4, 8]:
        data = {k: collect(t, N) for k, t in tags.items()}
        data = {k: v for k, v in data.items() if v}
        means = {k: round(float(np.mean(list(v.values()))), 4) for k, v in data.items()}
        cis = {k: boot_ci(list(v.values())) for k, v in data.items()}
        comps = []
        pairs = [("dixon5", "dixon4", "fat-fraction vs baseline"),
                 ("dixonfuse", "dixon4", "fat-fraction+gate vs baseline"),
                 ("dixonfuse", "dixon5", "fat-fraction+gate vs fat-fraction")]
        raw = []
        for a, b, lab in pairs:
            if a in data and b in data:
                da, db = paired(data[a], data[b])
                t, p = stats.ttest_rel(da, db)
                raw.append((lab, round(float((da - db).mean()), 4), float(p)))
        if raw:
            adj = holm([r[2] for r in raw])
            comps = [{"comparison": r[0], "delta": r[1], "p": round(r[2], 4),
                      "p_holm": round(float(a), 4)} for r, a in zip(raw, adj)]
        out[f"N{N}"] = {"n_subjects": len(next(iter(data.values()))),
                        "means": means, "ci95": cis, "comparisons": comps}
    json.dump(out, open(os.path.join(R, "dixonfuse_cv.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
