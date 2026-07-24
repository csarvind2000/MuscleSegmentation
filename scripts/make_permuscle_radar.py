"""Per-muscle Dice radar (spider) comparing the segmentation architectures on the thigh
(full training set, per-subject Dice). Regenerated for the revision to include the
U-Net+HRNet-W18 backbone. Reads results/persubj_thigh_<model>.json (from eval_persubject.py).
Writes figures/fig_permuscle_radar.(png|pdf).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

MODELS = [("unet_r34", "U-Net", "#1f77b4"),
          ("unetpp_r34", "U-Net++", "#2ca02c"),
          ("deeplabv3p_r34", "DeepLabV3+", "#ff7f0e"),
          ("segformer_b1", "SegFormer", "#9467bd"),
          ("unet_hrnet18", "U-Net+HRNet-W18", "#d62728")]

# display order (grouped), with short labels
ORDER = ["rectus_femoris", "vastus_lateralis", "vastus_intermedius", "vastus_medialis",
         "sartorius", "gracilis", "adductor_longus", "adductor_brevis", "adductor_magnus",
         "gluteus_maximus", "biceps_femoris", "semitendinosus", "semimembranosus"]
SHORT = {"rectus_femoris": "RF", "vastus_lateralis": "VL", "vastus_intermedius": "VI",
         "vastus_medialis": "VM", "sartorius": "SA", "gracilis": "GR",
         "adductor_longus": "AL", "adductor_brevis": "AB", "adductor_magnus": "AM",
         "gluteus_maximus": "GM", "biceps_femoris": "BF", "semitendinosus": "ST",
         "semimembranosus": "SM"}
# full anatomical names (two words wrapped onto two lines for the radar axis)
FULL = {"rectus_femoris": "Rectus\nfemoris", "vastus_lateralis": "Vastus\nlateralis",
        "vastus_intermedius": "Vastus\nintermedius", "vastus_medialis": "Vastus\nmedialis",
        "sartorius": "Sartorius", "gracilis": "Gracilis",
        "adductor_longus": "Adductor\nlongus", "adductor_brevis": "Adductor\nbrevis",
        "adductor_magnus": "Adductor\nmagnus", "gluteus_maximus": "Gluteus\nmaximus",
        "biceps_femoris": "Biceps\nfemoris", "semitendinosus": "Semitendinosus",
        "semimembranosus": "Semimembranosus"}

# Published 3D Attention-Res-V-Net per-muscle Dice (Wohlfarth et al., same dataset, 5-fold
# CV, 2-channel input) -- shown for reference; protocol differs from ours (see caption).
REF3D = {"rectus_femoris": 0.877, "vastus_lateralis": 0.891, "vastus_intermedius": 0.820,
         "vastus_medialis": 0.875, "sartorius": 0.871, "gracilis": 0.856,
         "adductor_longus": 0.857, "adductor_brevis": 0.761, "adductor_magnus": 0.900,
         "gluteus_maximus": 0.881, "biceps_femoris": 0.900, "semitendinosus": 0.822,
         "semimembranosus": 0.832}


def load(model):
    d = json.load(open(os.path.join(C.RESULTS, f"persubj_thigh_{model}.json")))["per_muscle"]
    return [d[m] for m in ORDER]


def main():
    N = len(ORDER)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(8.8, 8.8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    for key, name, col in MODELS:
        vals = load(key); vals += vals[:1]
        lw = 2.4 if key == "unet_hrnet18" else 1.6
        ax.plot(ang, vals, color=col, linewidth=lw, label=name)
        ax.fill(ang, vals, color=col, alpha=0.06)
    # published 3D reference as a distinct dashed floor polygon
    ref = [REF3D[m] for m in ORDER]; ref += ref[:1]
    ax.plot(ang, ref, color="#111111", linewidth=2.2, linestyle=(0, (5, 3)),
            label="3D Attention-Res-V-Net (ref.)", zorder=1)
    ax.fill(ang, ref, color="#111111", alpha=0.05)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([FULL[m] for m in ORDER], fontsize=10)
    ax.set_ylim(0.74, 0.965)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    ax.set_yticklabels(["0.75", "0.80", "0.85", "0.90", "0.95"], fontsize=8, color="#555")
    ax.tick_params(pad=14)
    ax.grid(color="#cccccc", linewidth=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=9, frameon=False)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG, f"fig_permuscle_radar.{ext}"),
                    dpi=150, bbox_inches="tight")
    means = {name: round(float(np.mean(load(key))), 4) for key, name, _ in MODELS}
    means["3D ref"] = round(float(np.mean([REF3D[m] for m in ORDER])), 4)
    print("wrote fig_permuscle_radar; means:", means)


if __name__ == "__main__":
    main()
