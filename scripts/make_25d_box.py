"""Figure: 2.5D vs 2D on the thigh, shown as a grouped box plot instead of a table.
For each architecture and each adjacent-slice context (2D k=0, 2.5D +/-1, 2.5D +/-2), the
box is the distribution of per-muscle slice-level Dice over the 13 thigh muscles.
Reads results/thigh_25d_k{0,1,2}_<model>.json; writes figures/fig_25d_box.(png|pdf).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import config as C

ARCHS = [("unet_r34", "U-Net"), ("unetpp_r34", "U-Net++"),
         ("deeplabv3p_r34", "DeepLabV3+"), ("segformer_b1", "SegFormer"),
         ("unet_hrnet18", "HRNet-W18")]
CTX = [(0, "2D ($k{=}0$)", "#9aa5b1"),
       (1, "2.5D ($\\pm1$)", "#1f77b4"),
       (2, "2.5D ($\\pm2$)", "#ff7f0e")]


def permuscle(k, model):
    f = os.path.join(C.RESULTS, f"thigh_25d_k{k}_{model}.json")
    return [v for v in json.load(open(f))["test_dice_per_class"][1:] if v is not None]


def main():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    width = 0.24
    xbase = np.arange(len(ARCHS))
    for j, (k, _, col) in enumerate(CTX):
        data = [permuscle(k, m) for m, _ in ARCHS]
        pos = xbase + (j - 1) * (width + 0.02)
        bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True,
                        showmeans=True, manage_ticks=False,
                        medianprops=dict(color="#111827", lw=1.4),
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="#111827", markersize=5),
                        flierprops=dict(marker="o", markersize=3.5,
                                        markerfacecolor=col, markeredgecolor="none", alpha=0.6))
        for patch in bp["boxes"]:
            patch.set_facecolor(col); patch.set_alpha(0.45); patch.set_edgecolor(col)

    ax.set_xticks(xbase)
    ax.set_xticklabels([lbl for _, lbl in ARCHS], fontsize=11)
    ax.set_ylabel("Per-muscle Dice (slice-level)", fontsize=12)
    ax.set_title("2.5D vs 2D on the thigh: distribution over 13 muscles", fontsize=13)
    ax.set_ylim(0.66, 0.96)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[Patch(facecolor=c, alpha=0.45, edgecolor=c, label=l)
                       for _, l, c in CTX], fontsize=10, loc="lower right",
              ncol=3, frameon=True)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG, f"fig_25d_box.{ext}"), dpi=180, bbox_inches="tight")
    plt.close()
    # report medians for the caption
    med = {lbl: {m: round(float(np.median(permuscle(k, m))), 3) for m, _ in ARCHS}
           for k, lbl, _ in CTX}
    print("wrote fig_25d_box; medians:", med)


if __name__ == "__main__":
    main()
