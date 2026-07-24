"""Figure 2: example data for all THREE datasets, each shown as image | ground truth.
Row 1 AATTCT-IDS (CT + SAT/VAT), Row 2 lower-limb Dixon MRI thigh (water + muscles),
Row 3 external multi-ethnic thigh (water + muscles). Writes figures/dataset_overview.(png)
"""
import os, glob, json, csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import config as C

AAT = mcolors.ListedColormap([[0, 0, 0], [0.16, 0.55, 1.0], [1.0, 0.32, 0.24]])


def muscle_cmap(n, seed=7):
    rng = np.random.RandomState(seed); c = rng.uniform(0.25, 1, (n, 3)); c[0] = 0
    return mcolors.ListedColormap(c)


def ct_pair():
    rows = [r for r in csv.DictReader(open(os.path.join(C.CACHE, "aattct_index.csv"))) if r["split"] == "test"]
    best = None
    for r in rows[::7][:40]:
        if (np.array(Image.open(r["vat"]).convert("L")) > 127).sum() > 6000:
            best = r; break
    r = best or rows[len(rows) // 2]
    g = np.array(Image.open(r["image"]).convert("L")).astype(float) / 255
    sat = np.array(Image.open(r["sat"]).convert("L")) > 127
    vat = np.array(Image.open(r["vat"]).convert("L")) > 127
    ov = np.stack([g, g, g], -1); ov[sat] = [0.16, 0.55, 1.0]; ov[vat] = [1.0, 0.32, 0.24]
    return g, np.clip(ov, 0, 1)


def mri_pair(cache, meta_file):
    meta = json.load(open(os.path.join(C.CACHE, meta_file)))
    subj = meta["subjects_test"][0]
    files = sorted(glob.glob(os.path.join(C.CACHE, cache, f"{subj}_*.npz")))
    d = np.load(files[len(files) // 2]); w = d["img"][0]; mask = d["mask"]
    w = (w - w.min()) / (np.ptp(w) + 1e-6)
    nc = int(mask.max()) + 1; over = muscle_cmap(max(nc, 12))(mask / max(nc - 1, 1))[..., :3]
    base = np.stack([w, w, w], -1); m = mask > 0
    seg = base.copy(); seg[m] = 0.4 * base[m] + 0.6 * over[m]
    return np.clip(w, 0, 1), np.clip(seg, 0, 1)


def grid(rows, out, figsize):
    fig, ax = plt.subplots(len(rows), 2, figsize=figsize, squeeze=False)
    for r, (name, img, seg, cm, gtlab) in enumerate(rows):
        ax[r, 0].imshow(img, cmap=cm); ax[r, 1].imshow(seg)
        ax[r, 0].set_ylabel(name, fontsize=9, rotation=90, labelpad=6)
        if r == 0:
            ax[r, 0].set_title("Image", fontsize=10)
            ax[r, 1].set_title("Ground truth", fontsize=10)
        ax[r, 1].text(0.5, -0.06, gtlab, transform=ax[r, 1].transAxes, ha="center",
                      va="top", fontsize=7.5, color="#444")
        for c in (0, 1):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(C.FIG, out), dpi=300, bbox_inches="tight")
    plt.close(); print("wrote", out)


def main():
    ct_i, ct_g = ct_pair()
    th_i, th_g = mri_pair("thigh_slices5", "thigh5_meta.json")
    ex_i, ex_g = mri_pair("ext_slices3", "ext_meta.json")
    # overview: the two PRIMARY datasets only (both unilateral / single-thigh)
    grid([("AATTCT-IDS\n(abdominal CT)", ct_i, ct_g, "gray", "SAT / VAT"),
          ("Lower-limb MRI\n(thigh, primary)", th_i, th_g, "gray", "13 muscles")],
         "dataset_overview.png", (5.0, 5.0))
    # external cohort shown separately (bilateral / both thighs, 11 muscles)
    grid([("Han-Chinese MRI\n(thigh, external)", ex_i, ex_g, "gray", "11 muscles, bilateral")],
         "dataset_external.png", (5.0, 2.7))


if __name__ == "__main__":
    main()
