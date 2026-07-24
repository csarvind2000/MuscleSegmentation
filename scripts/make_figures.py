"""Generate qualitative + quantitative figures from trained checkpoints.

Produces (under figures/):
  dataset_overview.png       - sample CT+adipose and MRI+muscle with masks
  qual_aattct.png            - image | GT | prediction (best model), a few test cases
  qual_thigh.png             - image | GT | prediction (best model), a few test cases
  perclass_dice.png          - grouped bar chart of per-class test Dice
Run AFTER training. Best model chosen by results/*.json dice_fg_mean.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import torch

import config as C
from models import build_model
from datasets import AATTCTDataset, ThighSliceDataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"

AAT_CMAP = mcolors.ListedColormap([[0, 0, 0], [0.15, 0.6, 1.0], [1.0, 0.3, 0.2]])  # bg, SAT, VAT


def best_model(task):
    cands = []
    for f in glob.glob(os.path.join(C.RESULTS, f"{task}_*.json")):
        r = json.load(open(f))
        cands.append((r["test_dice_fg_mean"], r["model"], r))
    cands.sort(reverse=True)
    return cands[0][1], cands[0][2]


def load_net(task, model_name, in_ch, nc):
    net = build_model(model_name, in_ch, nc)
    ck = os.path.join(C.CKPT, f"{task}_{model_name}.pt")
    net.load_state_dict(torch.load(ck, map_location=DEV))
    return net.to(DEV).eval()


def rand_cmap(n, seed=1):
    rng = np.random.RandomState(seed)
    cols = rng.rand(n, 3); cols[0] = 0
    return mcolors.ListedColormap(cols)


@torch.no_grad()
def qual_methods(n=3):
    """Same slices segmented by every architecture, side by side, for visual comparison."""
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    split = json.load(open(os.path.join(C.CACHE, "thigh_split.json")))
    ds = ThighSliceDataset(split["test"])
    nc = meta["num_classes"]
    cmap = rand_cmap(nc)
    archs = [("unet_r34", "U-Net"), ("unetpp_r34", "U-Net++"),
             ("deeplabv3p_r34", "DeepLabV3+"), ("segformer_b1", "SegFormer")]
    nets = [(lbl, load_net("thigh", m, meta["in_channels"], nc)) for m, lbl in archs]
    cols = ["MRI (water)", "Ground truth"] + [lbl for lbl, _ in nets]
    picks = np.linspace(len(ds) * 0.15, len(ds) * 0.85, n).astype(int)
    fig, ax = plt.subplots(n, len(cols), figsize=(2.1 * len(cols), 2.1 * n))
    for r, i in enumerate(picks):
        img, mask = ds[i]
        mask = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        w = img[0].numpy() if hasattr(img[0], "numpy") else np.asarray(img[0])
        lo, hi = np.percentile(w, [2, 98])  # window out the reflection so fat reads dark (true water look)
        ax[r, 0].imshow(w, cmap="gray", vmin=lo, vmax=hi)
        ax[r, 1].imshow(mask, cmap=cmap, vmin=0, vmax=nc - 1)
        for c, (lbl, net) in enumerate(nets):
            pred = net(img[None].to(DEV)).argmax(1)[0].cpu().numpy()
            ax[r, c + 2].imshow(pred, cmap=cmap, vmin=0, vmax=nc - 1)
        for c in range(len(cols)):
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(C.FIG, "qual_methods.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote qual_methods.png")


@torch.no_grad()
def qual_factors(subjects=("10",), seed=0):
    """One row per method (experimental factor) for each subject: water MRI, ground truth,
    then every variant of that factor side by side. Panels are cropped tight to the thigh
    so there is no wasted black margin between them."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from models import build_model
    from train_lowdata import build as build_lowdata
    from models_method import DixonFuseUNet
    from train_multitask import Branch
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    nc = meta["num_classes"]; cmap = rand_cmap(nc)
    c5 = os.path.join(C.CACHE, "thigh_slices5")

    def load(builder, ckpt):
        net = builder(); net.load_state_dict(torch.load(os.path.join(C.CKPT, ckpt), map_location=DEV))
        return net.to(DEV).eval()

    # (method label, [(variant label, builder, ckpt, uses_5ch), ...])
    methods = [
        ("Factor 1\narchitecture\n(full data)", [
            ("U-Net", lambda: build_model("unet_r34", 4, nc), "thigh_unet_r34.pt", False),
            ("U-Net++", lambda: build_model("unetpp_r34", 4, nc), "thigh_unetpp_r34.pt", False),
            ("DeepLabV3+", lambda: build_model("deeplabv3p_r34", 4, nc), "thigh_deeplabv3p_r34.pt", False),
            ("SegFormer", lambda: build_model("segformer_b1", 4, nc), "thigh_segformer_b1.pt", False)]),
        ("Factor 2\npretraining\n(n=2)", [
            ("random", lambda: build_lowdata("random", 4, nc, "thigh"), f"lowdata_random_n2_s{seed}.pt", False),
            ("ImageNet", lambda: build_lowdata("imagenet", 4, nc, "thigh"), f"lowdata_imagenet_n2_s{seed}.pt", False),
            ("CT-derived", lambda: build_lowdata("ct", 4, nc, "thigh"), f"lowdata_ct_n2_s{seed}.pt", False)]),
        ("Factor 3\nfat-fraction\n(n=2)", [
            ("no FF", lambda: DixonFuseUNet("dixon4", nc), f"method_dixon4_n2_s{seed}.pt", True),
            ("+FF", lambda: DixonFuseUNet("dixon5", nc), f"method_dixon5_n2_s{seed}.pt", True),
            ("+FF+gate", lambda: DixonFuseUNet("dixonfuse", nc), f"method_dixonfuse_n2_s{seed}.pt", True)]),
        ("Factor 4\nshared encoder\n(full data)", [
            ("single-task", lambda: build_model("unet_r34", 4, nc), "thigh_unet_r34.pt", False),
            ("shared", lambda: Branch(4, nc), "mtl_mri.pt", False)]),
    ]
    nets = [(lab, [(vl, load(b, ck), f5) for vl, b, ck, f5 in vs]) for lab, vs in methods]

    # per subject: slice with the most muscle classes, and a tight crop box around the thigh
    picks, crops = {}, {}
    for sid in subjects:
        fs = sorted(glob.glob(os.path.join(c5, f"{sid}_*.npz")))
        best = max(fs, key=lambda f: len(np.unique(np.load(f)["mask"])))
        picks[sid] = best
        m = np.load(best)["mask"]; ys, xs = np.where(m > 0); pad = 18
        crops[sid] = (max(ys.min() - pad, 0), min(ys.max() + pad, m.shape[0]),
                      max(xs.min() - pad, 0), min(xs.max() + pad, m.shape[1]))

    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    names = meta["classes"][1:]                        # 13 muscle names
    ncol = 2 + max(len(vs) for _, vs in nets)          # water + GT + widest variant set
    nmeth = len(nets); nsub = len(subjects)

    # row plan: subject block (nmeth rows), a spacer between subjects
    plan, hr = [], []
    for si in range(nsub):
        if si:
            plan.append(("sp", None, None)); hr.append(0.45)
        for mi in range(nmeth):
            plan.append(("m", si, mi)); hr.append(1.0)

    scale = 2.7 if nsub == 1 else 1.8              # bigger panels when a single subject
    fig = plt.figure(figsize=(scale * ncol, scale * 1.03 * sum(hr)))
    gs = GridSpec(len(plan), ncol, figure=fig, height_ratios=hr, hspace=0.26, wspace=0.04)

    # precompute predictions per subject
    cache = {}
    for sid in subjects:
        b5 = np.load(picks[sid]); img5 = b5["img"]; y0, y1, x0, x1 = crops[sid]
        crop = lambda a: a[y0:y1, x0:x1]
        x4 = torch.from_numpy(img5[:4][None]).float().to(DEV)
        x5 = torch.from_numpy(img5[None]).float().to(DEV)
        w = crop(img5[0]); lo, hi = np.percentile(w, [2, 98])
        preds = {}
        for mlabel, variants in nets:
            for vlabel, net, f5 in variants:
                preds[(mlabel, vlabel)] = crop(net(x5 if f5 else x4).argmax(1)[0].cpu().numpy())
        cache[sid] = (w, lo, hi, crop(b5["mask"]), preds)

    header_y = []
    for ri, (kind, si, mi) in enumerate(plan):
        if kind != "m":
            continue
        sid = subjects[si]; w, lo, hi, gt, preds = cache[sid]
        mlabel, variants = nets[mi]
        panels = [("water MRI", w, "gray"), ("ground truth", gt, cmap)] + \
                 [(vl, preds[(mlabel, vl)], cmap) for vl, _, _ in variants]
        for c, (ttl, arr, cm) in enumerate(panels):
            a = fig.add_subplot(gs[ri, c])
            if cm == "gray":
                a.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
            else:
                a.imshow(arr, cmap=cm, vmin=0, vmax=nc - 1)
            a.set_title(ttl, fontsize=9, fontweight="bold"); a.axis("off")
            if c == 0:
                a.text(-0.32, 0.5, mlabel, transform=a.transAxes, rotation=90,
                       va="center", ha="center", fontsize=9, fontweight="bold")
                if mi == 0:
                    header_y.append(a)

    fig.canvas.draw()
    if nsub > 1:                                    # subject bands only when >1 subject
        for si, a in enumerate(header_y):
            y = a.get_position().y1
            fig.text(0.5, min(y + 0.010, 0.998), f"Subject {chr(65 + si)}",
                     ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(C.FIG, "qual_factors.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote qual_factors.png (subjects", subjects, "slices",
          {k: os.path.basename(v) for k, v in picks.items()}, ")")


@torch.no_grad()
def qual_aattct(n=4):
    idx = os.path.join(C.CACHE, "aattct_index.csv")
    ds = AATTCTDataset(idx, "test")
    mname, _ = best_model("aattct")
    net = load_net("aattct", mname, 1, C.AATTCT_NUM_CLASSES)
    picks = np.linspace(0, len(ds) - 1, n).astype(int)
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    for r, i in enumerate(picks):
        img, mask = ds[i]
        pred = net(img[None].to(DEV)).argmax(1)[0].cpu().numpy()
        ax[r, 0].imshow(img[0], cmap="gray"); ax[r, 0].set_title("CT" if r == 0 else "")
        ax[r, 1].imshow(mask, cmap=AAT_CMAP, vmin=0, vmax=2); ax[r, 1].set_title("Ground truth" if r == 0 else "")
        ax[r, 2].imshow(pred, cmap=AAT_CMAP, vmin=0, vmax=2); ax[r, 2].set_title("Prediction" if r == 0 else "")
        for c in range(3): ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG, "qual_aattct.png"), dpi=150); plt.close()
    print("wrote qual_aattct.png (model:", mname, ")")


@torch.no_grad()
def qual_thigh(n=4):
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    split = json.load(open(os.path.join(C.CACHE, "thigh_split.json")))
    ds = ThighSliceDataset(split["test"])
    mname, _ = best_model("thigh")
    net = load_net("thigh", mname, meta["in_channels"], meta["num_classes"])
    cmap = rand_cmap(meta["num_classes"])
    picks = np.linspace(0, len(ds) - 1, n).astype(int)
    titles = ["MRI (water)", "Ground truth", "Prediction", "Error (GT ≠ pred)"]
    fig, ax = plt.subplots(n, 4, figsize=(12, 3 * n))
    for r, i in enumerate(picks):
        img, mask = ds[i]
        mask = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        pred = net(img[None].to(DEV)).argmax(1)[0].cpu().numpy()
        err = (pred != mask).astype(float)  # 1 where prediction disagrees with GT
        ax[r, 0].imshow(img[0], cmap="gray")
        ax[r, 1].imshow(mask, cmap=cmap, vmin=0, vmax=meta["num_classes"] - 1)
        ax[r, 2].imshow(pred, cmap=cmap, vmin=0, vmax=meta["num_classes"] - 1)
        ax[r, 3].imshow(img[0], cmap="gray"); ax[r, 3].imshow(np.ma.masked_where(err == 0, err), cmap="autumn_r", alpha=0.9, vmin=0, vmax=1)
        for c in range(4):
            ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(titles[c], fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG, "qual_thigh.png"), dpi=150); plt.close()
    print("wrote qual_thigh.png (model:", mname, ")")


def perclass_bars():
    label = {"aattct": "AATTCT-IDS (CT): SAT / VAT", "thigh": "Thigh MRI: 13 muscles"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for ax, task in zip(axes, ["aattct", "thigh"]):
        files = sorted(glob.glob(os.path.join(C.RESULTS, f"{task}_*.json")))
        if not files: continue
        rs = [json.load(open(f)) for f in files]
        classes = [c.replace("_", " ") for c in rs[0]["classes"][1:]]  # drop background
        x = np.arange(len(classes)); w = 0.8 / len(rs)
        for k, r in enumerate(rs):
            vals = [v if v is not None else 0 for v in r["test_dice_per_class"][1:]]
            ax.bar(x + k * w, vals, w, label=r["model"])
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=11)
        ax.set_ylabel("Dice", fontsize=12); ax.set_title(label.get(task, task), fontsize=13)
        ax.legend(fontsize=10, ncol=4, loc="lower center"); ax.set_ylim(0, 1.05)
        ax.tick_params(axis="y", labelsize=11); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG, "perclass_dice.png"), dpi=200); plt.close()
    print("wrote perclass_dice.png")


def dataset_overview():
    idx = os.path.join(C.CACHE, "aattct_index.csv")
    a = AATTCTDataset(idx, "test")
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    split = json.load(open(os.path.join(C.CACHE, "thigh_split.json")))
    t = ThighSliceDataset(split["test"])
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    ai, am = a[len(a) // 2]
    ax[0, 0].imshow(ai[0], cmap="gray"); ax[0, 0].set_title("AATTCT-IDS: CT")
    ax[0, 1].imshow(am, cmap=AAT_CMAP, vmin=0, vmax=2); ax[0, 1].set_title("SAT (blue) + VAT (red)")
    ti, tm = t[len(t) // 2]
    names = ["water", "fat", "in-phase", "opp-phase"]
    ax[0, 2].imshow(ti[0], cmap="gray"); ax[0, 2].set_title("Thigh MRI: water")
    ax[0, 3].imshow(ti[1], cmap="gray"); ax[0, 3].set_title("Thigh MRI: fat")
    ax[1, 0].imshow(ti[2], cmap="gray"); ax[1, 0].set_title("in-phase")
    ax[1, 1].imshow(ti[3], cmap="gray"); ax[1, 1].set_title("opp-phase")
    ax[1, 2].imshow(rand := tm, cmap=rand_cmap(meta["num_classes"]), vmin=0, vmax=meta["num_classes"] - 1)
    ax[1, 2].set_title("13 thigh muscles")
    ax[1, 3].axis("off")
    for r in range(2):
        for c in range(4):
            if not (r == 1 and c == 3): ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG, "dataset_overview.png"), dpi=150); plt.close()
    print("wrote dataset_overview.png")


if __name__ == "__main__":
    dataset_overview()
    qual_aattct()
    qual_thigh()
    perclass_bars()
