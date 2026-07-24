"""Clinical volumetric agreement for thigh-muscle segmentation.

For each held-out test subject, run the trained DixonFuse model at native
resolution, compute per-muscle volume (mL) from voxel counts and the NIfTI voxel
spacing, and compare with ground truth. Reports:
  - per-muscle volume MAE (mL) and relative volume error (%)
  - intraclass correlation (ICC) for predicted vs true volume
  - Bland-Altman bias and 95% limits of agreement (total muscle volume)
  - mean fat-fraction error within each muscle
  - error stratified by small vs large muscles
Writes results/volumetrics.json and figures/bland_altman.png.
"""
import os, glob, json
import numpy as np
import nibabel as nib
from skimage.transform import resize as _rs
import torch
import config as C
from models_method import DixonFuseUNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONTRASTS = ["Water", "Fat", "In_phase", "Opp_phase"]


def z(v): return (v - v.mean()) / (v.std() + 1e-6)


def load_subject(subj, region="Thigh"):
    d = os.path.join(C.THIGH_ROOT, subj, region)
    nii = nib.load(os.path.join(d, "Water.nii.gz"))
    vols = {c: nib.load(os.path.join(d, c + ".nii.gz")).get_fdata() for c in CONTRASTS}
    m = nib.load(os.path.join(d, "mask_muscles.nii.gz")).get_fdata().astype(np.int64)
    vox_ml = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0  # mm^3 -> mL
    return vols, m, vox_ml


@torch.no_grad()
def predict_native(model, vols, labels, remap):
    W, F = vols["Water"], vols["Fat"]
    ff = np.clip(F / (F + W + 1e-6), 0, 1)
    ch = [z(vols["Water"]), z(vols["Fat"]), z(vols["In_phase"]), z(vols["Opp_phase"]), ff]
    H, Wd, D = W.shape
    pred = np.zeros((H, Wd, D), dtype=np.int64)
    for k in range(D):
        img = np.stack([_rs(c[:, :, k], (256, 256), order=1, preserve_range=True) for c in ch], 0).astype(np.float32)
        logits = model(torch.from_numpy(img)[None].to(DEV))
        p = logits.argmax(1)[0].cpu().numpy()
        pred[:, :, k] = _rs(p, (H, Wd), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
    return pred


def icc(x, y):
    # ICC(2,1) absolute agreement, two-way random
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x); m = np.vstack([x, y]).T
    grand = m.mean()
    ms_r = 2 * ((m.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((m.mean(0) - grand) ** 2).sum() / 1
    ms_e = (((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + grand) ** 2).sum()) / (n - 1)
    return float((ms_r - ms_e) / (ms_r + ms_e + (2 / n) * (ms_c - ms_e) + 1e-9))


def main():
    meta = json.load(open(os.path.join(C.CACHE, "thigh5_meta.json")))
    labels = meta["orig_labels"]; names = meta["classes"]; nc = meta["num_classes"]
    remap = {int(v): i for i, v in enumerate(labels)}
    model = DixonFuseUNet("dixonfuse", nc).to(DEV)
    model.load_state_dict(torch.load(os.path.join(C.CKPT, "method_dixonfuse_n18_s0.pt"), map_location=DEV))
    model.eval()

    rows = []  # (subj, class_idx, true_ml, pred_ml, ff_true, ff_pred)
    for s in meta["subjects_test"]:
        vols, mask, vox_ml = load_subject(s)
        mask_r = np.vectorize(remap.get)(mask).astype(np.int64)
        pred = predict_native(model, vols, labels, remap)
        W, F = vols["Water"], vols["Fat"]; ff = np.clip(F / (F + W + 1e-6), 0, 1)
        for ci in range(1, nc):
            tvol = float((mask_r == ci).sum() * vox_ml)
            pvol = float((pred == ci).sum() * vox_ml)
            ff_t = float(ff[mask_r == ci].mean()) if (mask_r == ci).any() else np.nan
            ff_p = float(ff[pred == ci].mean()) if (pred == ci).any() else np.nan
            rows.append((s, ci, tvol, pvol, ff_t, ff_p))
        print(f"{s}: done", flush=True)

    arr = np.array([(r[2], r[3], r[4], r[5]) for r in rows], float)
    cls = np.array([r[1] for r in rows])
    tv, pv, fft, ffp = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    abs_err = np.abs(pv - tv)
    rel_err = 100 * abs_err / (tv + 1e-6)
    # per-muscle aggregates
    per_muscle = {}
    for ci in range(1, nc):
        mmask = cls == ci
        per_muscle[names[ci]] = {
            "true_ml_mean": round(float(tv[mmask].mean()), 2),
            "vol_mae_ml": round(float(abs_err[mmask].mean()), 2),
            "vol_rel_err_pct": round(float(rel_err[mmask].mean()), 1),
            "ff_abs_err": round(float(np.nanmean(np.abs(ffp[mmask] - fft[mmask]))), 4),
        }
    # total muscle volume per subject (Bland-Altman)
    subs = meta["subjects_test"]
    tot_t = [sum(r[2] for r in rows if r[0] == s) for s in subs]
    tot_p = [sum(r[3] for r in rows if r[0] == s) for s in subs]
    diff = np.array(tot_p) - np.array(tot_t)
    bias = float(diff.mean()); loa = 1.96 * float(diff.std(ddof=1))
    # size stratification (median split on true mean volume)
    sizes = {ci: tv[cls == ci].mean() for ci in range(1, nc)}
    med = np.median(list(sizes.values()))
    small = [names[ci] for ci in sizes if sizes[ci] < med]
    small_mask = np.isin([names[c] for c in cls], small)
    strat = {"small_muscles_rel_err_pct": round(float(rel_err[small_mask].mean()), 1),
             "large_muscles_rel_err_pct": round(float(rel_err[~small_mask].mean()), 1)}

    out = {"model": "DixonFuse (ImageNet, N=18)",
           "n_test_subjects": len(subs),
           "overall_vol_mae_ml": round(float(abs_err.mean()), 2),
           "overall_vol_rel_err_pct": round(float(rel_err.mean()), 1),
           "icc_volume": round(icc(tv, pv), 3),
           "total_muscle_bland_altman": {"bias_ml": round(bias, 1), "loa95_ml": round(loa, 1)},
           "ff_abs_err_mean": round(float(np.nanmean(np.abs(ffp - fft))), 4),
           "size_stratified": strat,
           "per_muscle": per_muscle}
    json.dump(out, open(os.path.join(C.RESULTS, "volumetrics.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

    # Bland-Altman figure
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mean_tp = (np.array(tot_p) + np.array(tot_t)) / 2
    plt.figure(figsize=(5, 4))
    plt.scatter(mean_tp, diff, c="#2b6cb0")
    plt.axhline(bias, color="k"); plt.axhline(bias + loa, color="r", ls="--"); plt.axhline(bias - loa, color="r", ls="--")
    plt.xlabel("Mean of predicted and true total muscle volume (mL)")
    plt.ylabel("Predicted - true (mL)"); plt.title("Bland-Altman: total thigh muscle volume")
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG, "bland_altman.png"), dpi=160); plt.close()
    print("wrote volumetrics.json and bland_altman.png")


if __name__ == "__main__":
    main()
