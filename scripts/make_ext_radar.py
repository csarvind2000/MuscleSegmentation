"""External-cohort per-muscle Dice radar (spider). Uses the headline external config
(ImageNet + Water/Fat/FF, n=46, 3-seed mean) from ext_permuscle_wff_imagenet_n46.json.

Label -> muscle mapping is the order supplied for the HuashanMyo cohort
(label 1..11 = BL, BB, ST, SM, AM, VI, VL, VM, RF, GR, SA).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

# label(1..11) -> short muscle code, confirmed from the HuashanMyo label editor (ITK-SNAP)
NAMES = ["SA", "RF", "VL", "VI", "VM", "AM", "GR", "BL", "ST", "SM", "BB"]
FULL = {"BL": "Biceps fem. long", "BB": "Biceps fem. short", "ST": "Semitendinosus",
        "SM": "Semimembranosus", "AM": "Adductor magnus", "VI": "Vastus interm.",
        "VL": "Vastus lat.", "VM": "Vastus med.", "RF": "Rectus femoris",
        "GR": "Gracilis", "SA": "Sartorius"}
# draw order: group by compartment (quadriceps, hamstrings, adductor/medial) for readability
ORDER = ["RF", "VL", "VI", "VM", "BL", "BB", "ST", "SM", "AM", "GR", "SA"]


def main():
    d = json.load(open(os.path.join(C.RESULTS, "ext_permuscle_wff_imagenet_n46.json")))
    dice = {NAMES[i - 1]: d["per_class_dice_mean"][i] for i in range(1, d["num_classes"])}
    vals = [dice[m] for m in ORDER]
    labels = [f"{m}\n{v:.2f}" for m, v in zip(ORDER, vals)]

    ang = np.linspace(0, 2 * np.pi, len(ORDER), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    val_c = vals + vals[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.plot(ang_c, val_c, "-o", color="#1f77b4", lw=2, ms=5)
    ax.fill(ang_c, val_c, color="#1f77b4", alpha=0.18)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=10)
    rmin = 0.80
    ax.set_ylim(rmin, 0.90)
    ax.set_yticks([0.82, 0.84, 0.86, 0.88])
    ax.set_yticklabels(["0.82", "0.84", "0.86", "0.88"], fontsize=8, color="gray")
    ax.set_title("External cohort: per-muscle Dice\n(ImageNet + Water/Fat/FF, n=46)",
                 fontsize=12, pad=18)
    plt.tight_layout()
    fig.savefig(os.path.join(C.FIG, "fig_ext_radar.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(C.FIG, "fig_ext_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote fig_ext_radar; muscle:dice =", {m: round(dice[m], 3) for m in ORDER})


def overlay():
    """Two-polygon radar: primary (internal) vs external cohort on the shared muscles,
    to show the domain gap per muscle."""
    ext = json.load(open(os.path.join(C.RESULTS, "ext_permuscle_wff_imagenet_n46.json")))
    ext_d = {NAMES[i - 1]: ext["per_class_dice_mean"][i] for i in range(1, ext["num_classes"])}
    itn = json.load(open(os.path.join(C.RESULTS, "internal_permuscle_unetpp.json")))
    # shared muscle code -> internal (full) name
    shared = {"RF": "rectus_femoris", "VL": "vastus_lateralis", "VI": "vastus_intermedius",
              "VM": "vastus_medialis", "ST": "semitendinosus", "SM": "semimembranosus",
              "AM": "adductor_magnus", "GR": "gracilis", "SA": "sartorius"}
    order = ["RF", "VL", "VI", "VM", "ST", "SM", "AM", "GR", "SA"]   # quads, hams, medial
    iv = [itn[shared[m]] for m in order]
    ev = [ext_d[m] for m in order]
    labels = [f"{m}\n$\\Delta${itn[shared[m]] - ext_d[m]:+.02f}" for m in order]

    ang = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    for vals, col, lab in [(iv, "#d62728", "Primary cohort (U-Net++, full data)"),
                           (ev, "#1f77b4", "External cohort (ImageNet+WFF, full data)")]:
        vc = vals + vals[:1]
        ax.plot(ang_c, vc, "-o", color=col, lw=2, ms=5, label=lab)
        ax.fill(ang_c, vc, color=col, alpha=0.12)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0.80, 0.97)
    ax.set_yticks([0.83, 0.87, 0.91, 0.95])
    ax.set_yticklabels(["0.83", "0.87", "0.91", "0.95"], fontsize=8, color="gray")
    ax.set_title("Per-muscle Dice: primary vs external cohort\n(shared muscles; "
                 "$\\Delta$ = primary $-$ external)", fontsize=12, pad=20)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(C.FIG, "fig_domaingap_radar.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(C.FIG, "fig_domaingap_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    gaps = {m: round(itn[shared[m]] - ext_d[m], 3) for m in order}
    print("wrote fig_domaingap_radar; per-muscle gap (primary-external) =", gaps)


if __name__ == "__main__":
    main()
    overlay()
