"""Professional manuscript architecture diagrams with embedded real CT/MRI images.
  figures/fig_study_design.(png|pdf)  - overview: datasets(images) -> 4 axes -> eval
  figures/fig_dixonfuse.(png|pdf)     - DixonFuse method with real image flow
"""
import os, glob, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import colors as mcolors
import config as C

# ---- palette (muted, professional) ----
CBG   = "#f1f5f9"
CCT   = "#dbeafe"; CCT_E = "#3b82f6"
CMRI  = "#ffe4cc"; CMRI_E = "#f97316"
CAXIS = "#e7f0ea"; CAX_E = "#2f6b4f"
CMETH = "#fdf0c9"; CMET_E = "#c99700"
CEVAL = "#efe6fb"; CEV_E = "#7c3aed"
INK   = "#0f172a"; GREY = "#475569"; LINE = "#334155"

AAT_CMAP = mcolors.ListedColormap([[0, 0, 0], [0.16, 0.55, 1.0], [1.0, 0.32, 0.24]])


def muscle_cmap(n, seed=7):
    rng = np.random.RandomState(seed)
    cols = rng.uniform(0.25, 1.0, (n, 3)); cols[0] = 0
    return mcolors.ListedColormap(cols)


def _round(patch_xy, w, h, fc, ec, lw=1.4, rad=0.03):
    return FancyBboxPatch(patch_xy, w, h,
                          boxstyle=f"round,pad=0.015,rounding_size={rad}",
                          linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)


def box(ax, x, y, w, h, title, sub, fc, ec, tfs=10, sfs=8.3):
    ax.add_patch(_round((x, y), w, h, fc, ec))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=INK, zorder=5)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sfs, fontweight="bold", color=INK, zorder=5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=INK, zorder=5)


def arrow(ax, p1, p2, color=LINE, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=15,
                                 lw=lw, color=color, shrinkA=3, shrinkB=3, zorder=2))


# ---------- load real sample images ----------
def ct_overlay():
    idx = os.path.join(C.CACHE, "aattct_index.csv")
    import csv
    rows = [r for r in csv.DictReader(open(idx)) if r["split"] == "test"]
    # pick a slice with substantial visceral fat
    best = None
    for r in rows[::7][:40]:
        v = np.array(Image.open(r["vat"]).convert("L")) > 127
        if v.sum() > 6000:
            best = r; break
    r = best or rows[len(rows) // 2]
    g = np.array(Image.open(r["image"]).convert("L")).astype(float) / 255
    sat = np.array(Image.open(r["sat"]).convert("L")) > 127
    vat = np.array(Image.open(r["vat"]).convert("L")) > 127
    rgb = np.stack([g, g, g], -1)
    rgb[sat] = [0.16, 0.55, 1.0]
    rgb[vat] = [1.0, 0.32, 0.24]
    return np.clip(rgb, 0, 1)


def mri_sample():
    meta = json.load(open(os.path.join(C.CACHE, "thigh5_meta.json")))
    files = sorted(glob.glob(os.path.join(C.CACHE, "thigh_slices5", f"{meta['subjects_test'][0]}_*.npz")))
    d = np.load(files[len(files) // 2])
    water = d["img"][0]; fat = d["img"][1]; ff = d["img"][4]; mask = d["mask"]
    w = (water - water.min()) / (np.ptp(water) + 1e-6)
    nc = int(mask.max()) + 1
    cmap = muscle_cmap(max(nc, 14))
    over = cmap(mask / max(nc - 1, 1))[..., :3]
    base = np.stack([w, w, w], -1)
    m = mask > 0
    blend = base.copy(); blend[m] = 0.45 * base[m] + 0.55 * over[m]
    fatn = (fat - fat.min()) / (np.ptp(fat) + 1e-6)
    return np.clip(w, 0, 1), np.clip(blend, 0, 1), np.clip(ff, 0, 1), np.clip(fatn, 0, 1)


def inset(ax, x, y, w, h, img, title=None, cmap=None, ec="#94a3b8"):
    a = ax.inset_axes([x, y, w, h], transform=ax.transData)
    a.imshow(img, cmap=cmap); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_edgecolor(ec); s.set_linewidth(1.3)
    if title:
        a.set_title(title, fontsize=8.8, color=INK, pad=2, fontweight="bold")
    return a


# ============================ FIGURE 1 ============================
def study_design():
    ct = ct_overlay()
    w_img, mri_over, ff_img, _ = mri_sample()
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.text(6.75, 7.32, "Systematic study of data-scarce body-composition segmentation",
            ha="center", fontsize=14.5, fontweight="bold", color=INK)

    for cx, t in [(2.35, "TWO PRIMARY DATASETS"), (7.0, "FOUR EXPERIMENTAL FACTORS"), (11.55, "EVALUATION")]:
        ax.text(cx, 6.75, t, ha="center", fontsize=12, fontweight="bold", color=GREY)

    # ---- data: image + caption ----
    inset(ax, 0.35, 4.55, 1.5, 1.5, ct, "CT + SAT/VAT", ec=CCT_E)
    box(ax, 2.05, 4.75, 2.6, 1.15, "AATTCT-IDS (CT)",
        "300 subjects  ·  3213 slices\n512×512  ·  SAT/VAT (3 cls)", CCT, CCT_E, tfs=10.0, sfs=8.2)
    inset(ax, 0.35, 2.35, 1.5, 1.5, mri_over, "MRI + 13 muscles", ec=CMRI_E)
    box(ax, 2.05, 2.55, 2.6, 1.15, "Thigh Dixon MRI",
        "25 subjects  ·  1513 slices\n256×256  ·  4ch  ·  14 cls", CMRI, CMRI_E, tfs=10.0, sfs=8.2)
    box(ax, 0.35, 1.15, 4.05, 0.85, "Subject-level split  70 / 10 / 20",
        "no slice leakage across sets", CBG, "#94a3b8", tfs=10.0, sfs=8.2)

    # ---- axes ----
    axes = [
        ("Factor 1 . Architecture", "U-Net | U-Net++ | DeepLabV3+ | SegFormer", CAXIS, CAX_E, 5.35),
        ("Factor 2 . Pretraining source", "random | ImageNet | CT | self-supervised (SSL)", CAXIS, CAX_E, 4.15),
        ("Factor 3 . Input design", "+fat-fraction (primary)  ·  +SE gate (ablation)", CMETH, CMET_E, 2.95),
        ("Factor 4 . Multi-task learning", "shared encoder, two task decoders", CAXIS, CAX_E, 1.75),
    ]
    for title, sub, fc, ec, y in axes:
        box(ax, 5.15, y, 3.75, 1.0, title, sub, fc, ec, tfs=10.0, sfs=8.2)

    # ---- eval ----
    box(ax, 9.95, 4.95, 3.2, 1.05, "Per-subject Dice & IoU", "held-out subjects", CEVAL, CEV_E, tfs=10.0, sfs=8.4)
    box(ax, 9.95, 3.6, 3.2, 1.05, "Data-efficiency + 5-fold CV", "N=2,4,8,18; per-subject", CEVAL, CEV_E, tfs=10.0, sfs=8.4)
    box(ax, 9.95, 2.25, 3.2, 1.05, "3 seeds -> mean +/- std", "channel-weight & occlusion", CEVAL, CEV_E, tfs=10.0, sfs=8.4)

    # ---- arrows ----
    arrow(ax, (4.7, 5.3), (5.15, 5.55)); arrow(ax, (4.7, 5.1), (5.15, 4.55))
    arrow(ax, (4.7, 3.1), (5.15, 3.35)); arrow(ax, (4.7, 3.0), (5.15, 2.15))
    for y in (5.85, 4.65, 3.45, 2.25):
        arrow(ax, (8.9, y), (9.95, 5.4 if y > 4.5 else (4.1 if y > 3 else 2.8)))

    ax.add_patch(_round((0.3, 0.12), 12.9, 0.92, "#f8fafc", "#cbd5e1", lw=1.2, rad=0.02))
    ax.text(6.75, 0.76,
            "Findings:   (1) architecture differences comparatively small     "
            "(2) ImageNet and CT initialisation statistically equivalent",
            ha="center", va="center", fontsize=9.4, fontweight="bold", color=LINE)
    ax.text(6.75, 0.40,
            "(3) fat-fraction proxy: significant gain at the lowest labels     "
            "(4) shared encoder close to single-task",
            ha="center", va="center", fontsize=9.4, fontweight="bold", color=LINE)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG, f"fig_study_design.{ext}"), dpi=220, bbox_inches="tight")
    plt.close()
    print("wrote fig_study_design.(png|pdf)")


# ============================ FIGURE 2 ============================
def dixonfuse():
    w_img, mri_over, ff_img, fat_img = mri_sample()
    fig, ax = plt.subplots(figsize=(13.5, 4.4))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(6.75, 4.12, "DixonFuse: physics-informed contrast fusion for thigh-muscle MRI",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)

    y, h = 1.35, 1.25
    # input contrasts (mini stack)
    inset(ax, 0.35, 1.15, 1.35, 1.55, w_img, "Water", cmap="gray", ec=CMRI_E)
    ax.text(1.02, 0.92, "Dixon 4ch (W,F,In,Opp)", ha="center", fontsize=7.8, color=GREY)
    # FF channel
    inset(ax, 2.5, 1.15, 1.35, 1.55, ff_img, "Fat-fraction  FF=F/(F+W)", cmap="magma", ec=CMET_E)
    box(ax, 4.5, y, 1.7, h, "Contrast\nAttention", "SE gate x5", CMETH, CMET_E, tfs=9.4, sfs=7.6)
    box(ax, 6.55, y, 1.35, h, "1x1 conv", "5 -> 3 ch", CBG, "#94a3b8", tfs=9.4, sfs=7.6)
    box(ax, 8.25, y, 1.9, h, "ResNet-34\nencoder", "ImageNet", CAXIS, CAX_E, tfs=9.4, sfs=7.6)
    box(ax, 10.5, y, 1.35, h, "U-Net\ndecoder", "", CAXIS, CAX_E, tfs=9.4, sfs=7.6)
    inset(ax, 12.05, 1.15, 1.15, 1.55, mri_over, "14-class mask", ec=CMRI_E)

    for p1, p2 in [((1.7, 1.9), (2.5, 1.9)), ((3.85, 1.9), (4.5, 1.97)),
                   ((6.2, 1.97), (6.55, 1.97)), ((7.9, 1.97), (8.25, 1.97)),
                   ((10.15, 1.97), (10.5, 1.97)), ((11.85, 1.97), (12.05, 1.9))]:
        arrow(ax, p1, p2)
    ax.text(6.75, 0.42, "learned per-contrast gate weights are reported (interpretability)",
            ha="center", fontsize=8, style="italic", color=GREY)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG, f"fig_dixonfuse.{ext}"), dpi=220, bbox_inches="tight")
    plt.close()
    print("wrote fig_dixonfuse.(png|pdf)")


# ============================ WORKFLOW ============================
def workflow():
    w_img, mri_over, ff_img, _ = mri_sample()
    fig, ax = plt.subplots(figsize=(13.5, 3.0))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 3.0); ax.axis("off")
    ax.text(6.75, 2.78, "Label-efficient individual-muscle segmentation workflow",
            ha="center", fontsize=13, fontweight="bold", color=INK)
    y, h = 0.95, 1.15
    inset(ax, 0.30, 0.78, 1.25, 1.45, w_img, "Dixon MRI\n(W,F,In,Opp)", cmap="gray", ec=CMRI_E)
    stages = [
        (1.95, 2.05, "Preprocessing", "standardise, resize,\nfat-fraction proxy", CBG, "#94a3b8"),
        (4.25, 2.25, "Label-efficient\nmodel", "ImageNet init +\nfat-fraction input", CAXIS, CAX_E),
        (6.75, 2.15, "Individual-muscle\nmasks", "13 thigh muscles", CMETH, CMET_E),
        (9.15, 2.35, "Volume & fat-\nfraction measures", "per-muscle mL, FF;\nICC, Bland-Altman", CEVAL, CEV_E),
    ]
    for x, wbox, title, sub, fc, ec in stages:
        box(ax, x, y, wbox, h, title, sub, fc, ec, tfs=9.9, sfs=8.4)
    box(ax, 11.75, y, 1.5, h, "Structured\noutput", "NIfTI masks +\nCSV table", CBG, "#94a3b8", tfs=9.6, sfs=8.2)
    xs = [1.55, 1.95 + 2.05, 4.25 + 2.25, 6.75 + 2.15, 9.15 + 2.35]
    tgt = [1.95, 4.25, 6.75, 9.15, 11.75]
    for a, b in zip(xs, tgt):
        arrow(ax, (a, 1.52), (b, 1.52))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG, f"fig_workflow.{ext}"), dpi=220, bbox_inches="tight")
    plt.close()
    print("wrote fig_workflow.(png|pdf)")


if __name__ == "__main__":
    study_design()
    dixonfuse()
    workflow()
