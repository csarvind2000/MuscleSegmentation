"""Results figures for the manuscript:
  fig_transfer.(png|pdf)   - pretraining source: Dice vs #subjects (random/ImageNet/CT)
  fig_method.(png|pdf)     - DixonFuse variants: Dice vs #subjects
Both read results/*.json (3-seed mean +/- std).
"""
import os, glob, re, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
NS = [2, 4, 8, 18]


def collect(pat_str, key):
    pat = re.compile(pat_str); d = defaultdict(list)
    for f in glob.glob(os.path.join(C.RESULTS, key)):
        m = pat.search(os.path.basename(f))
        if m:
            d[(m.group(1), int(m.group(2)))].append(json.load(open(f))["test_dice_fg_mean"])
    return d


def curve(ax, d, series, title, xlabel="# thigh-MRI training subjects"):
    for name, lab, col in series:
        xs, ys, es = [], [], []
        for n in NS:
            v = d.get((name, n))
            if v: xs.append(n); ys.append(np.mean(v)); es.append(np.std(v))
        if not xs: continue
        ys, es = np.array(ys), np.array(es)
        ax.plot(xs, ys, "-o", color=col, label=lab, lw=2, ms=6)
        ax.fill_between(xs, ys - es, ys + es, color=col, alpha=0.15)
    ax.set_xticks(NS); ax.set_xlabel(xlabel); ax.set_ylabel("Mean foreground Dice")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)


def main():
    dt = collect(r"lowdata_(\w+?)_n(\d+)_s(\d+)\.json", "lowdata_*.json")
    dm = collect(r"method_(\w+?)_n(\d+)_s(\d+)\.json", "method_*.json")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    curve(ax, dt, [("random", "Random init", "#888"),
                   ("imagenet", "ImageNet init", "#1f77b4"),
                   ("ct", "CT-pretrained", "#d62728")],
          "Encoder pretraining source")
    plt.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(C.FIG, f"fig_transfer.{e}"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    curve(ax, dm, [("dixon4", "Dixon four-channel baseline", "#888"),
                   ("dixon5", "Fat-fraction", "#2ca02c"),
                   ("dixonfuse", "Fat-fraction and contrast gate", "#d62728")],
          "Physics-guided input design")
    plt.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(C.FIG, f"fig_method.{e}"), dpi=200)
    plt.close()
    print("wrote fig_transfer.(png|pdf) and fig_method.(png|pdf)")


if __name__ == "__main__":
    main()
