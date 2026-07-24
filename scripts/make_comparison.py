"""Two-panel bar chart of the two decisive factors, read directly from the
cross-validation results so the values match Tables tab:cv and tab:cvmethod exactly.
(a) encoder pretraining at N=8; (b) input design at N=2. Writes figures/fig_comparison.pdf.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import config as C


def stat(pat):
    acc = defaultdict(list)
    for f in glob.glob(os.path.join(C.RESULTS, pat)):
        for s, v in json.load(open(f))["per_subject_dice"].items():
            acc[s].append(v)
    a = np.array([np.mean(v) for v in acc.values()])
    return a.mean(), a.std()


def fmt(m):                       # round-half-up to 3 dp so labels match the tables
    return format(m + 5e-10, ".3f")


def main():
    panels = [
        ("(a) Encoder pretraining ($N{=}8$)", [
            ("Random", "cv_thigh_init-random_n8_f*_r*.json", "#9e9e9e"),
            ("CT", "cv_thigh_init-ct_n8_f*_r*.json", "#4c9f70"),
            ("ImageNet", "cv_thigh_init-imagenet_n8_f*_r*.json", "#1f77b4")]),
        ("(b) Input design ($N{=}2$)", [
            ("4-channel", "cv_thigh_method-dixon4_n2_f*_r*.json", "#9e9e9e"),
            ("gate-only", "cv_thigh_method-gateonly_n2_f*_r*.json", "#c48f2a"),
            ("$+$FF", "cv_thigh_method-dixon5_n2_f*_r*.json", "#1f77b4"),
            ("$+$FF$+$gate", "cv_thigh_method-dixonfuse_n2_f*_r*.json", "#6a5acd")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (title, bars) in zip(axes, panels):
        labels = [b[0] for b in bars]
        means = [stat(b[1])[0] for b in bars]
        sds = [stat(b[1])[1] for b in bars]
        cols = [b[2] for b in bars]
        x = np.arange(len(bars))
        ax.bar(x, means, 0.62, yerr=sds, color=cols, capsize=4,
               error_kw=dict(lw=1, ecolor="#444"))
        for xi, m in zip(x, means):
            ax.text(xi, m + 0.005, fmt(m), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Per-subject Dice", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.78, 0.93)
    axes[1].set_ylim(0.60, 0.83)
    plt.tight_layout()
    fig.savefig(os.path.join(C.FIG, "fig_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(C.FIG, "fig_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote fig_comparison; (a)",
          [f"{b[0]}={fmt(stat(b[1])[0])}" for b in panels[0][1]],
          "(b)", [f"{b[0]}={fmt(stat(b[1])[0])}" for b in panels[1][1]])


if __name__ == "__main__":
    main()
