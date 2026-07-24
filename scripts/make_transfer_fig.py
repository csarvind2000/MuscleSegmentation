"""Headline figure + table for the cross-modality transfer / data-efficiency study.

Reads results/lowdata_<init>_n<N>_s<seed>.json and produces:
  figures/transfer_curve.png  - Dice vs #MRI training subjects, one line per init
  results/transfer_table.csv  - mean +/- std foreground Dice per (init, N)
"""
import os, glob, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

INITS = ["random", "imagenet", "ct"]
LABELS = {"random": "Random init", "imagenet": "ImageNet init", "ct": "CT-pretrained (ours)"}
COLORS = {"random": "#888888", "imagenet": "#1f77b4", "ct": "#d62728"}
PAT = re.compile(r"lowdata_(\w+?)_n(\d+)_s(\d+)\.json")


def collect():
    data = {}  # (init, n) -> list of dice
    for f in glob.glob(os.path.join(C.RESULTS, "lowdata_*.json")):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        init, n, s = m.group(1), int(m.group(2)), int(m.group(3))
        r = json.load(open(f))
        data.setdefault((init, n), []).append(r["test_dice_fg_mean"])
    return data


def main():
    data = collect()
    ns = sorted({n for (_, n) in data})
    plt.figure(figsize=(6, 4.2))
    rows = [["init", "n_subjects", "n_seeds", "dice_mean", "dice_std"]]
    for init in INITS:
        xs, ys, es = [], [], []
        for n in ns:
            vals = data.get((init, n))
            if not vals:
                continue
            xs.append(n); ys.append(np.mean(vals)); es.append(np.std(vals))
            rows.append([init, n, len(vals), round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)])
        if not xs:
            continue
        ys, es = np.array(ys), np.array(es)
        plt.plot(xs, ys, "-o", color=COLORS[init], label=LABELS[init])
        plt.fill_between(xs, ys - es, ys + es, color=COLORS[init], alpha=0.15)
    plt.xlabel("# thigh-MRI training subjects")
    plt.ylabel("Mean foreground Dice (test)")
    plt.title("Cross-modality transfer for thigh-muscle segmentation")
    plt.xticks(ns); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    out = os.path.join(C.FIG, "transfer_curve.png")
    plt.savefig(out, dpi=160); plt.close()
    open(os.path.join(C.RESULTS, "transfer_table.csv"), "w").write("\n".join(",".join(map(str, r)) for r in rows))
    print("wrote", out)
    print("\n".join(",".join(map(str, r)) for r in rows))


if __name__ == "__main__":
    main()
