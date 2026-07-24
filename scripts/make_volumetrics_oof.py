"""Out-of-fold volumetric agreement over ALL 25 thigh subjects (reviewer item 5).

Uses the same fixed 5-fold partition as cross_val.py (repeat 0). For each fold,
trains a fat-fraction segmenter (dixon5 = Water,Fat,In,Opp,FF, no gate) on the other
20 subjects and predicts the 5 held-out subjects at native resolution. Every subject
is therefore predicted by a model that never saw it -> 25 unbiased volume estimates.

Reports total-muscle-volume Bland-Altman (bias, LoA, 95% CI), ICC(2,1) with CI for
total volume, and per-muscle ICC (median/range). Writes results/volumetrics_oof.json
and figures/bland_altman.png (25 points).
"""
import os, glob, json, random, time
import numpy as np
import nibabel as nib
from skimage.transform import resize as _rs
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import config as C
from datasets import ThighSliceDataset
from models_method import DixonFuseUNet
from train import DiceCELoss
from cross_val import folds, sfiles

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONTRASTS = ["Water", "Fat", "In_phase", "Opp_phase"]
VARIANT = "dixon5"  # fat-fraction input, no gate (recommended configuration)


def z(v): return (v - v.mean()) / (v.std() + 1e-6)


def load_subject(subj, region="Thigh"):
    d = os.path.join(C.THIGH_ROOT, subj, region)
    nii = nib.load(os.path.join(d, "Water.nii.gz"))
    vols = {c: nib.load(os.path.join(d, c + ".nii.gz")).get_fdata() for c in CONTRASTS}
    m = nib.load(os.path.join(d, "mask_muscles.nii.gz")).get_fdata().astype(np.int64)
    vox_ml = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
    return vols, m, vox_ml


@torch.no_grad()
def predict_native(model, vols):
    W, F = vols["Water"], vols["Fat"]
    ff = np.clip(F / (F + W + 1e-6), 0, 1)
    ch = [z(vols["Water"]), z(vols["Fat"]), z(vols["In_phase"]), z(vols["Opp_phase"]), ff]
    H, Wd, D = W.shape
    pred = np.zeros((H, Wd, D), dtype=np.int64)
    for k in range(D):
        img = np.stack([_rs(c[:, :, k], (256, 256), order=1, preserve_range=True) for c in ch], 0).astype(np.float32)
        p = model(torch.from_numpy(img)[None].to(DEV)).argmax(1)[0].cpu().numpy()
        pred[:, :, k] = _rs(p, (H, Wd), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
    return pred, ff


def train_fold(train_subjects, nc, epochs=70):
    cache = os.path.join(C.CACHE, "thigh_slices5")
    tr = DataLoader(ThighSliceDataset(sfiles(train_subjects, cache), augment=True), 8, True,
                    num_workers=6, pin_memory=True, drop_last=True)
    model = DixonFuseUNet(VARIANT, nc).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = DiceCELoss(nc).to(DEV); scaler = torch.cuda.amp.GradScaler()
    for ep in range(epochs):
        model.train()
        for img, mask in tr:
            img, mask = img.to(DEV), mask.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    model.eval(); return model


def icc21_point(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measure."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x); k = 2; m = np.vstack([x, y]).T
    grand = m.mean()
    ms_r = k * ((m.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((m.mean(0) - grand) ** 2).sum() / (k - 1)
    ms_e = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n + 1e-12))


def icc21(x, y, nboot=5000):
    """ICC(2,1) with a subject-clustered percentile bootstrap 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    val = icc21_point(x, y)
    n = len(x); rng = np.random.RandomState(0); bs = []
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        try:
            bs.append(icc21_point(x[idx], y[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(val), float(lo), float(hi)


def main():
    meta = json.load(open(os.path.join(C.CACHE, "thigh5_meta.json")))
    labels = meta["orig_labels"]; names = meta["classes"]; nc = meta["num_classes"]
    remap = {int(v): i for i, v in enumerate(labels)}
    fs = folds(0)  # same partition as CV repeat 0
    all_subs = [f"{i:02d}" for i in range(1, 26)]
    t0 = time.time(); rows = []
    for fi, test in enumerate(fs):
        train = sorted([s for s in all_subs if s not in test])
        random.seed(fi); torch.manual_seed(fi); np.random.seed(fi)
        model = train_fold(train, nc)
        for s in test:
            vols, mask, vox_ml = load_subject(s)
            mask_r = np.vectorize(remap.get)(mask).astype(np.int64)
            pred, ff = predict_native(model, vols)
            for ci in range(1, nc):
                tm = (mask_r == ci); pm = (pred == ci)
                rows.append((s, ci, float(tm.sum() * vox_ml), float(pm.sum() * vox_ml),
                             float(ff[tm].mean()) if tm.any() else np.nan,
                             float(ff[pm].mean()) if pm.any() else np.nan))
        print(f"fold {fi}: {test} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    cls = np.array([r[1] for r in rows])
    tv = np.array([r[2] for r in rows]); pv = np.array([r[3] for r in rows])
    fft = np.array([r[4] for r in rows]); ffp = np.array([r[5] for r in rows])
    abs_err = np.abs(pv - tv); rel_err = 100 * abs_err / (tv + 1e-6)
    subs = all_subs
    tot_t = np.array([sum(r[2] for r in rows if r[0] == s) for s in subs])
    tot_p = np.array([sum(r[3] for r in rows if r[0] == s) for s in subs])
    diff = tot_p - tot_t
    bias = float(diff.mean()); sd = float(diff.std(ddof=1)); loa = 1.96 * sd
    n = len(subs)
    se_bias = sd / np.sqrt(n); se_loa = np.sqrt(3) * sd / np.sqrt(n)
    from scipy import stats
    tcrit = stats.t.ppf(0.975, n - 1)
    icc_tot, icc_lo, icc_hi = icc21(tot_t, tot_p)
    # size stratification (median split on mean true muscle volume)
    sizes = {ci: tv[cls == ci].mean() for ci in range(1, nc)}
    med = np.median(list(sizes.values()))
    small = {ci for ci in sizes if sizes[ci] < med}
    small_mask = np.isin(cls, list(small))
    # per-muscle
    permu = {}
    for ci in range(1, nc):
        mm = cls == ci
        val, lo, hi = icc21(tv[mm], pv[mm])
        permu[names[ci]] = {"icc": round(val, 3), "ci": [round(lo, 3), round(hi, 3)],
                            "true_ml_mean": round(float(tv[mm].mean()), 1),
                            "vol_mae_ml": round(float(abs_err[mm].mean()), 2),
                            "vol_rel_err_pct": round(float(rel_err[mm].mean()), 1)}
    iccs = [permu[k]["icc"] for k in permu]
    out = {"model": f"{VARIANT} (fat-fraction, out-of-fold, N=25)",
           "n_subjects": n,
           "overall_vol_mae_ml": round(float(abs_err.mean()), 2),
           "overall_vol_rel_err_pct": round(float(rel_err.mean()), 1),
           "ff_abs_err_mean": round(float(np.nanmean(np.abs(ffp - fft))), 4),
           "size_stratified": {"small_muscles_rel_err_pct": round(float(rel_err[small_mask].mean()), 1),
                               "large_muscles_rel_err_pct": round(float(rel_err[~small_mask].mean()), 1)},
           "total_volume": {
               "icc21": round(icc_tot, 3), "icc21_ci": [round(icc_lo, 3), round(icc_hi, 3)],
               "bias_ml": round(bias, 1), "bias_ci": [round(bias - tcrit * se_bias, 1), round(bias + tcrit * se_bias, 1)],
               "loa_lower_ml": round(bias - loa, 1), "loa_upper_ml": round(bias + loa, 1),
               "loa_lower_ci": [round(bias - loa - tcrit * se_loa, 1), round(bias - loa + tcrit * se_loa, 1)],
               "loa_upper_ci": [round(bias + loa - tcrit * se_loa, 1), round(bias + loa + tcrit * se_loa, 1)]},
           "per_muscle_icc_median": round(float(np.median(iccs)), 3),
           "per_muscle_icc_range": [round(float(min(iccs)), 3), round(float(max(iccs)), 3)],
           "per_muscle": permu,
           "raw_rows": [[r[0], names[r[1]], round(r[2], 2), round(r[3], 2)] for r in rows]}
    json.dump(out, open(os.path.join(C.RESULTS, "volumetrics_oof.json"), "w"), indent=2)
    # CSV export of per-subject per-muscle volumes (the pipeline's structured output)
    import csv as _csv
    with open(os.path.join(C.RESULTS, "volumetrics_oof.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["subject", "muscle", "true_volume_mL", "pred_volume_mL",
                    "ff_proxy_true", "ff_proxy_pred"])
        for r in rows:
            w.writerow([r[0], names[r[1]], round(r[2], 2), round(r[3], 2),
                        round(r[4], 4) if r[4] == r[4] else "", round(r[5], 4) if r[5] == r[5] else ""])
    print(json.dumps(out, indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pear = float(stats.pearsonr(tot_t, tot_p)[0])
    mean_tp = (tot_p + tot_t) / 2
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    lo = min(tot_t.min(), tot_p.min()) * 0.95; hi = max(tot_t.max(), tot_p.max()) * 1.05
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax[0].scatter(tot_t, tot_p, c="#2b6cb0", s=28, zorder=3)
    ax[0].set_xlabel("Reference total muscle volume (mL)")
    ax[0].set_ylabel("Predicted total muscle volume (mL)")
    ax[0].set_title(f"(a) Agreement  (ICC$_{{2,1}}$={icc_tot:.3f}, r={pear:.3f})", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=0.3)
    ax[1].scatter(mean_tp, diff, c="#2b6cb0", s=28, zorder=3)
    ax[1].axhline(bias, color="k", label=f"bias {bias:.0f} mL")
    ax[1].axhline(bias + loa, color="r", ls="--", label=f"95% LoA ±{loa:.0f} mL")
    ax[1].axhline(bias - loa, color="r", ls="--")
    ax[1].set_xlabel("Mean of predicted and reference (mL)")
    ax[1].set_ylabel("Predicted − reference (mL)")
    ax[1].set_title(f"(b) Bland–Altman  (n={n})", fontsize=10)
    ax[1].legend(fontsize=8, loc="best"); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.FIG, "bland_altman.png"), dpi=300); plt.close()
    print("wrote volumetrics_oof.json and bland_altman.png")


if __name__ == "__main__":
    main()
