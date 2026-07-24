"""Nature-style architecture figure for DixonFuse.
Panel a: full pipeline (real images -> fusion -> U-Net encoder/decoder with skips -> mask)
Panel b: contrast-attention (SE) module detail
Outputs figures/fig_architecture_nature.(png|pdf)  (vector, 300 dpi)
"""
import os, glob, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import colors as mcolors
import config as C

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.8, "svg.fonttype": "none",
})

ENC = "#cfe3f7"; ENC_E = "#2b6cb0"      # encoder blue
DEC = "#d7ecd9"; DEC_E = "#2f855a"      # decoder green
BOT = "#e9d8fd"; BOT_E = "#6b46c1"      # bottleneck purple
FUSE = "#fdecc8"; FUSE_E = "#b7791f"    # fusion amber
INK = "#1a202c"; GREY = "#4a5568"; LINE = "#2d3748"; SKIP = "#a0aec0"


def muscle_cmap(n, seed=7):
    rng = np.random.RandomState(seed); c = rng.uniform(0.25, 1, (n, 3)); c[0] = 0
    return mcolors.ListedColormap(c)


def load_imgs():
    meta = json.load(open(os.path.join(C.CACHE, "thigh5_meta.json")))
    f = sorted(glob.glob(os.path.join(C.CACHE, "thigh_slices5", f"{meta['subjects_test'][0]}_*.npz")))
    d = np.load(f[len(f) // 2]); water = d["img"][0]; ff = d["img"][4]; mask = d["mask"]
    w = (water - water.min()) / (np.ptp(water) + 1e-6)
    nc = int(mask.max()) + 1; over = muscle_cmap(max(nc, 14))(mask / max(nc - 1, 1))[..., :3]
    base = np.stack([w, w, w], -1); m = mask > 0
    seg = base.copy(); seg[m] = 0.4 * base[m] + 0.6 * over[m]
    return np.clip(w, 0, 1), np.clip(ff, 0, 1), np.clip(seg, 0, 1)


def img_at(ax, cx, cy, s, img, cmap=None, ec="#718096", label=None, lfs=8):
    a = ax.inset_axes([cx - s / 2, cy - s / 2, s, s], transform=ax.transData)
    a.imshow(img, cmap=cmap); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_edgecolor(ec); sp.set_linewidth(1.1)
    if label: a.set_title(label, fontsize=lfs, color=GREY, pad=2)


def fmap(ax, cx, top, spatial, chan, fc, ec, label_ch=True):
    """Stacked feature-map block: height ~ spatial, width ~ log2(channels)."""
    h = 0.22 + 1.9 * (spatial / 256.0)
    w = 0.16 + 0.075 * np.log2(chan)
    y = top - h
    for k in (2, 1, 0):  # depth shadow
        off = 0.05 * k
        ax.add_patch(Rectangle((cx - w / 2 + off, y + off), w, h, facecolor=fc,
                               edgecolor=ec, linewidth=1.0, zorder=3 - k,
                               alpha=1.0 if k == 0 else 0.55))
    if label_ch:
        ax.text(cx, y - 0.16, f"{chan}", ha="center", va="top", fontsize=7.4, color=ec, fontweight="bold")
        ax.text(cx, top + 0.08, f"{spatial}", ha="center", va="bottom", fontsize=6.6, color=GREY)
    return cx, y + h / 2, w, y, y + h


def arr(ax, p1, p2, color=LINE, lw=1.4, style="-|>", ms=11, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, lw=lw,
                                 color=color, shrinkA=1.5, shrinkB=1.5, zorder=6,
                                 linestyle=ls, connectionstyle="arc3,rad=0"))


def rbox(ax, x, y, w, h, t, sub, fc, ec, tfs=8.6, sfs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                                linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), t, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=INK, zorder=6)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=sfs, color=GREY, zorder=6)


def main():
    w_img, ff_img, seg = load_imgs()
    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.6); ax.axis("off")

    # panel labels
    ax.text(0.15, 6.42, "a", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.15, 2.18, "b", fontsize=15, fontweight="bold", color=INK)

    # ---------- panel a: inputs ----------
    img_at(ax, 0.95, 5.0, 1.0, w_img, cmap="gray", ec=FUSE_E, label="Water")
    img_at(ax, 0.95, 3.75, 1.0, ff_img, cmap="magma", ec=FUSE_E, label="Fat-fraction")
    ax.text(0.95, 3.1, "Dixon 4ch + FF\n(5 channels)", ha="center", va="top", fontsize=7.6, color=GREY)

    # fusion module box
    rbox(ax, 1.75, 3.95, 1.15, 1.35, "DixonFuse", "contrast\nattention\n+ 1x1 conv", FUSE, FUSE_E, tfs=8.4, sfs=7.0)
    arr(ax, (1.5, 4.55), (1.75, 4.62)); arr(ax, (1.5, 3.9), (1.75, 4.5))

    # ---------- U-Net encoder ----------
    tops = 5.95
    enc_specs = [(256, 64), (128, 64), (64, 128), (32, 256)]
    enc_x = [3.55, 4.35, 5.15, 5.95]
    enc_nodes = []
    for (sp, ch), cx in zip(enc_specs, enc_x):
        enc_nodes.append(fmap(ax, cx, tops, sp, ch, ENC, ENC_E))
    # bottleneck
    bcx = 6.75
    bx, bcy, bw, by0, by1 = fmap(ax, bcx, tops, 16, 512, BOT, BOT_E)
    # decoder (mirror)
    dec_specs = [(32, 256), (64, 128), (128, 64), (256, 32)]
    dec_x = [7.55, 8.35, 9.15, 9.95]
    dec_nodes = []
    for (sp, ch), cx in zip(dec_specs, dec_x):
        dec_nodes.append(fmap(ax, cx, tops, sp, ch, DEC, DEC_E))

    # encoder down arrows
    prev = (2.9, 4.62)  # from fusion box right edge
    arr(ax, prev, (enc_nodes[0][0] - enc_nodes[0][2] / 2 - 0.02, enc_nodes[0][1]))
    for a_, b_ in zip(enc_nodes[:-1], enc_nodes[1:]):
        arr(ax, (a_[0] + a_[2] / 2, a_[1]), (b_[0] - b_[2] / 2, b_[1]), color=ENC_E, lw=1.3)
    arr(ax, (enc_nodes[-1][0] + enc_nodes[-1][2] / 2, enc_nodes[-1][1]), (bcx - bw / 2, bcy), color=ENC_E, lw=1.3)
    # decoder up arrows
    arr(ax, (bcx + bw / 2, bcy), (dec_nodes[0][0] - dec_nodes[0][2] / 2, dec_nodes[0][1]), color=DEC_E, lw=1.3)
    for a_, b_ in zip(dec_nodes[:-1], dec_nodes[1:]):
        arr(ax, (a_[0] + a_[2] / 2, a_[1]), (b_[0] - b_[2] / 2, b_[1]), color=DEC_E, lw=1.3)

    # skip connections (encoder level -> mirrored decoder level)
    for e, d in zip(enc_nodes, dec_nodes[::-1]):
        ytop = max(e[4], d[4]) + 0.28
        ax.add_patch(FancyArrowPatch((e[0], e[4]), (d[0], d[4]),
                     arrowstyle="-|>", mutation_scale=9, lw=1.1, color=SKIP,
                     connectionstyle=f"arc3,rad=-0.35", zorder=2, linestyle=(0, (4, 2))))
    ax.text(6.75, 6.35, "skip connections", ha="center", fontsize=7.4, color=SKIP, style="italic")

    # output
    arr(ax, (dec_nodes[-1][0] + dec_nodes[-1][2] / 2, dec_nodes[-1][1]), (11.05, 4.9), color=DEC_E, lw=1.3)
    img_at(ax, 11.6, 4.9, 1.05, seg, ec=DEC_E, label="13-muscle mask")
    ax.text(11.6, 4.28, "softmax\n14 classes", ha="center", va="top", fontsize=7.2, color=GREY)

    # group labels
    ax.text((enc_x[0] + enc_x[-1]) / 2, 2.75, "Encoder  (ResNet-34, ImageNet)", ha="center", fontsize=8, color=ENC_E, fontweight="bold")
    ax.text(bcx, 2.75, "bottleneck", ha="center", fontsize=7.6, color=BOT_E, fontweight="bold")
    ax.text((dec_x[0] + dec_x[-1]) / 2, 2.75, "Decoder  (U-Net)", ha="center", fontsize=8, color=DEC_E, fontweight="bold")
    ax.plot([3.15, 6.35], [2.95, 2.95], color=ENC_E, lw=1.0)
    ax.plot([7.15, 10.35], [2.95, 2.95], color=DEC_E, lw=1.0)

    # ---------- panel b: SE attention detail ----------
    by = 0.55
    ax.text(0.55, 1.95, "Contrast-attention (squeeze-and-excitation over the 5 input contrasts)",
            fontsize=8.6, fontweight="bold", color=INK)
    # 5-channel input stack
    for i, name in enumerate(["W", "F", "In", "Op", "FF"]):
        ax.add_patch(Rectangle((0.55, by + i * 0.17), 0.55, 0.15, facecolor=FUSE, edgecolor=FUSE_E, lw=0.8, zorder=4))
        ax.text(0.82, by + i * 0.17 + 0.075, name, ha="center", va="center", fontsize=6.6, color=INK, zorder=5)
    ax.text(0.82, by - 0.12, "5 contrasts", ha="center", fontsize=7, color=GREY)
    seq = [("GAP", 1.6), ("FC\n5->2", 2.5), ("ReLU", 3.35), ("FC\n2->5", 4.2), ("sigmoid\ngate", 5.15)]
    prevx = 1.1
    for t, x in seq:
        rbox(ax, x, by, 0.7, 0.85, t.split("\n")[0], t.split("\n")[1] if "\n" in t else "", CBG_() , GREY, tfs=7.6, sfs=6.6)
        arr(ax, (prevx, by + 0.42), (x, by + 0.42), lw=1.1, ms=9); prevx = x + 0.7
    # scale
    ax.add_patch(FancyBboxPatch((6.05, by + 0.15), 0.55, 0.55, boxstyle="circle,pad=0.02",
                                linewidth=1.2, edgecolor=FUSE_E, facecolor=FUSE, zorder=4))
    ax.text(6.32, by + 0.42, "x", ha="center", va="center", fontsize=12, color=INK, zorder=6)
    arr(ax, (prevx, by + 0.42), (6.05, by + 0.42), lw=1.1, ms=9)
    arr(ax, (0.82, by + 5 * 0.17 + 0.02), (0.82, by + 1.15), lw=1.0, ms=8, color=SKIP, style="-|>", ls=(0, (3, 2)))
    arr(ax, (0.82, by + 1.2), (6.2, by + 0.75), lw=1.0, ms=8, color=SKIP, ls=(0, (3, 2)))
    ax.text(6.95, by + 0.42, "reweighted\ncontrasts", ha="left", va="center", fontsize=7.2, color=GREY)
    arr(ax, (6.6, by + 0.42), (6.9, by + 0.42), lw=1.1, ms=9)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG, f"fig_architecture_nature.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote fig_architecture_nature.(png|pdf)")


def CBG_():
    return "#edf2f7"


if __name__ == "__main__":
    main()
