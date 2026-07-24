"""Aggregate the reviewer-response experiments once they finish.

Cross-validation: pool per-subject Dice across folds/repeats, report mean and 95%
bootstrap CI per configuration, paired comparisons with Holm correction, and a TOST
equivalence test for ImageNet vs CT (margin 0.02 Dice).
Also summarises the MTL normalisation study, the gate-only ablation, and SSL.
Writes results/review_summary.json and prints a report.
"""
import os, glob, json, re
from collections import defaultdict
import numpy as np
from scipy import stats
import config as C
R = C.RESULTS


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = np.random.RandomState(0).randint(0, len(x), (n, len(x)))
    means = x[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cv_collect(model_tag, region="thigh", n=8):
    """Return dict subject -> list of dice across folds/repeats for a model tag."""
    per_subj = defaultdict(list)
    for f in glob.glob(os.path.join(R, f"cv_{region}_{model_tag}_n{n}_f*_r*.json")):
        d = json.load(open(f))
        for s, v in d["per_subject_dice"].items():
            per_subj[s].append(v)
    return per_subj


def paired(a, b):
    """Paired by subject (mean over repeats per subject). Returns diff array."""
    keys = sorted(set(a) & set(b))
    da = np.array([np.mean(a[k]) for k in keys])
    db = np.array([np.mean(b[k]) for k in keys])
    return da, db, keys


def tost(da, db, margin=0.02):
    d = da - db; n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    t_lower = (m + margin) / se; p_lower = 1 - stats.t.cdf(t_lower, n - 1)
    t_upper = (m - margin) / se; p_upper = stats.t.cdf(t_upper, n - 1)
    return max(p_lower, p_upper), float(m)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    prev = 0
    for rank, i in enumerate(order):
        adj[i] = max(prev, min(1.0, (m - rank) * pvals[i])); prev = adj[i]
    return adj


def main():
    out = {}
    # ---- cross-validation core claims ----
    cv = {}
    for tag in ["method-dixon4", "method-dixonfuse", "init-random", "init-imagenet", "init-ct"]:
        ps = cv_collect(tag)
        if not ps:
            continue
        vals = [np.mean(v) for v in ps.values()]
        lo, hi = boot_ci(vals)
        cv[tag] = {"n_subjects": len(vals), "mean": round(float(np.mean(vals)), 4),
                   "ci95": [round(lo, 4), round(hi, 4)], "per_subject": {k: round(float(np.mean(v)), 4) for k, v in ps.items()}}
    out["cross_validation"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_subject"} for k, v in cv.items()}

    # paired comparisons + Holm + TOST
    comps = []
    def add(a, b, label):
        if a in cv and b in cv:
            da, db, _ = paired(cv_collect(a), cv_collect(b))
            t, p = stats.ttest_rel(da, db)
            comps.append([label, round(float((da - db).mean()), 4), float(p)])
    add("method-dixonfuse", "method-dixon4", "DixonFuse vs Dixon4 (CV, N=8)")
    add("init-imagenet", "init-random", "ImageNet vs random (CV)")
    add("init-imagenet", "init-ct", "ImageNet vs CT (CV)")
    add("init-ct", "init-random", "CT vs random (CV)")
    if comps:
        adj = holm([c[2] for c in comps])
        out["comparisons"] = [{"test": c[0], "mean_diff": c[1], "p": round(c[2], 4),
                               "p_holm": round(float(a), 4)} for c, a in zip(comps, adj)]
    if "init-imagenet" in cv and "init-ct" in cv:
        da, db, _ = paired(cv_collect("init-imagenet"), cv_collect("init-ct"))
        p_tost, m = tost(da, db, 0.02)
        out["tost_imagenet_vs_ct"] = {"margin": 0.02, "mean_diff": round(m, 4),
                                      "p_tost": round(float(p_tost), 4),
                                      "equivalent": bool(p_tost < 0.05)}

    # ---- MTL normalisation ----
    norm = {}
    for k in ["shared_bn", "groupnorm", "instancenorm"]:
        fs = glob.glob(os.path.join(R, f"mtlnorm_{k}_s*.json"))
        if fs:
            ct = [json.load(open(f))["ct_dice_fg"] for f in fs]
            mri = [json.load(open(f))["mri_dice_fg"] for f in fs]
            norm[k] = {"ct": round(float(np.mean(ct)), 4), "mri": round(float(np.mean(mri)), 4), "seeds": len(fs)}
    out["mtl_norm"] = norm

    # ---- gate-only ablation ----
    def mmean(pat):
        vs = [json.load(open(f))["test_dice_fg_mean"] for f in glob.glob(os.path.join(R, pat))]
        return round(float(np.mean(vs)), 4) if vs else None
    out["gate_only"] = {f"n{n}": mmean(f"method_gateonly_n{n}_s*.json") for n in [2, 4, 8, 18]}
    out["ssl_downstream"] = {f"n{n}": mmean(f"lowdata_ssl_n{n}_s*.json") for n in [2, 4, 8, 18]}

    json.dump(out, open(os.path.join(R, "review_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
