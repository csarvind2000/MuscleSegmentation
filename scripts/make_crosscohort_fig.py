"""R1.1 revision figure: cross-cohort transfer and adaptation.

(a) TRUE zero-shot: primary-trained models applied directly to all 66 HuashanMyo
    subjects (no retraining). Bars = subject-level mean Dice per architecture; the
    dashed line marks in-domain primary performance for reference.
(b) Few-shot within-cohort adaptation: Dice vs number of target-cohort training
    subjects (N), by encoder initialisation. Zero-shot (N=0) is marked for context.

Reads results/ext_zeroshot_full.json and results/ext_{wff}_{init}_n{N}_s{seed}.json.
Writes manuscript_reviewed/fig_crosscohort.png
"""
import os, json, glob, re, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

OUT = "/media/ranjhaa-local/volume23/sarcopenia/seg_benchmark/manuscript_reviewed/fig_crosscohort.png"
PRIMARY_INDOMAIN = 0.855   # positive-control mean Dice on the primary thigh test set


def load_zeroshot():
    d = json.load(open(os.path.join(C.RESULTS, "ext_zeroshot_full.json")))
    h = json.load(open(os.path.join(C.RESULTS, "ext_hrnet_zs.json")))["unet_hrnet18"]
    order = ["unet_r34", "unetpp_r34", "deeplabv3p_r34", "segformer_b1"]
    names = ["U-Net", "U-Net++", "DeepLabV3+", "SegFormer-B1", "HRNet-W18"]
    means = [d[m]["overall_mean_dice"] for m in order] + [h["overall_mean_dice"]]
    sds = [d[m]["overall_std_dice"] for m in order] + [h["overall_std_dice"]]
    return names, means, sds, d["unet_r34"]["overall_mean_dice"]


def load_adapt():
    """Matched 4-channel / 10-muscle pipeline: primary fine-tune vs ImageNet-from-scratch."""
    agg = collections.defaultdict(list)
    for f in glob.glob(os.path.join(C.RESULTS, "ext_adapt_*_n*_s*.json")):
        m = re.match(r"ext_adapt_(primary|imagenet)_n(\d+)_s\d+\.json", os.path.basename(f))
        if not m:
            continue
        agg[(m.group(1), int(m.group(2)))].append(json.load(open(f))["test_dice_mean"])
    return agg


def load_labelfree():
    """U-Net label-free adaptation variants (means/SDs) and their honesty class."""
    dr = json.load(open(os.path.join(C.RESULTS, "ext_dr.json")))
    ab = json.load(open(os.path.join(C.RESULTS, "ext_adabn.json")))["adabn"]
    da = json.load(open(os.path.join(C.RESULTS, "ext_dr_adabn.json")))
    names = ["U-Net\n+ DomRand", "U-Net\n+ AdaBN", "U-Net\n+ DR+AdaBN"]
    means = [dr["mean"], ab["mean"], da["mean"]]
    sds = [dr["sd"], ab["sd"], da["sd"]]
    kind = ["zero", "sf", "sf"]                 # DomRand stays zero-shot; AdaBN uses target images
    return names, means, sds, kind


def main():
    znames, zmeans, zsds, unet_zs = load_zeroshot()
    lnames, lmeans, lsds, lkind = load_labelfree()
    agg = load_adapt()

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # ---- panel (a): zero-shot per architecture + U-Net label-free adaptation ----
    C_ZS, C_DR, C_SF = "#4C72B0", "#55A868", "#DD8452"
    xa = np.arange(len(znames))                        # architectures at 0..3
    xl = np.arange(len(lnames)) + len(znames) + 1      # label-free at 5..7 (gap at 4)
    ax[0].bar(xa, zmeans, yerr=zsds, capsize=3, color=C_ZS, width=0.62, alpha=0.9)
    lcolors = [C_DR if k == "zero" else C_SF for k in lkind]
    ax[0].bar(xl, lmeans, yerr=lsds, capsize=3, color=lcolors, width=0.62, alpha=0.9)
    ax[0].axvline(len(znames), color="#bbb", lw=0.8, ls=":")
    ax[0].axhline(PRIMARY_INDOMAIN, ls="--", color="#555", lw=1.2)
    ax[0].text(xl[-1], PRIMARY_INDOMAIN + 0.015, f"in-domain (primary) {PRIMARY_INDOMAIN:.2f}",
               ha="right", va="bottom", fontsize=8.5, color="#555")
    for xi, m in list(zip(xa, zmeans)) + list(zip(xl, lmeans)):
        ax[0].text(xi, m + 0.02, f"{m:.2f}", ha="center", fontsize=8.5)
    ax[0].set_xticks(np.concatenate([xa, xl]))
    ax[0].set_xticklabels(znames + lnames, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("Mean per-subject Dice")
    ax[0].set_ylim(0, 1.0)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=C_ZS, label="Zero-shot (per architecture)"),
                          Patch(color=C_DR, label="Zero-shot + domain randomisation"),
                          Patch(color=C_SF, label="Source-free adapt. (AdaBN; target images, no labels)")],
                 fontsize=7.3, loc="upper left", bbox_to_anchor=(0.005, 0.80), framealpha=0.95)
    ax[0].set_title("(a) Zero-shot and label-free adaptation: primary $\\rightarrow$ 66 HuashanMyo",
                    fontsize=9.5)

    # ---- panel (b): few-shot adaptation, matched pipeline (4ch, 10-muscle metric) ----
    Ns = [4, 8, 46]
    # primary line starts at the zero-shot point (N=0)
    py = [unet_zs] + [np.mean(agg[("primary", n)]) for n in Ns]
    pe = [zsds[0]] + [np.std(agg[("primary", n)]) for n in Ns]
    ax[1].errorbar([0] + Ns, py, yerr=pe, marker="o", color="#C44E52", capsize=3, lw=2.0,
                   label="Primary model (fine-tune)")
    iy = [np.mean(agg[("imagenet", n)]) for n in Ns]
    ie = [np.std(agg[("imagenet", n)]) for n in Ns]
    ax[1].errorbar(Ns, iy, yerr=ie, marker="s", color="#4C72B0", capsize=3, lw=2.0,
                   label="ImageNet-initialised target-only")
    ax[1].scatter([0], [unet_zs], color="k", zorder=6, marker="*", s=110)
    ax[1].annotate(f"zero-shot (N=0) {unet_zs:.2f}", (0, unet_zs), textcoords="offset points",
                   xytext=(10, -4), fontsize=8.5, va="center")
    # best label-free ceiling (DR+AdaBN, robust across 5 architectures) for reference
    BEST_LABELFREE = 0.30
    ax[1].axhline(BEST_LABELFREE, ls="--", color="#55A868", lw=1.4, zorder=2,
                  label="Best label-free ceiling (DR+AdaBN, $\\approx0.30$)")
    ax[1].text(43, BEST_LABELFREE + 0.012, "best label-free ($\\approx0.30$)",
               ha="right", va="bottom", fontsize=8, color="#3d7a4e")
    ax[1].set_xlabel("Target-cohort training subjects ($N$)")
    ax[1].set_ylabel("Mean per-subject Dice (10 shared muscles)")
    ax[1].set_xticks([0] + Ns)
    ax[1].set_ylim(0, 0.95)
    ax[1].legend(fontsize=8.0, loc="lower right")
    ax[1].set_title("(b) Closing the gap: source-model fine-tuning\n(train on $N$ target subjects, test on held-out)", fontsize=10)

    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight", metadata={"Software": None})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
