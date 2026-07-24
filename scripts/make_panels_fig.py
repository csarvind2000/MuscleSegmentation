"""Clean per-axis architecture panels (one simple schematic per experimental axis).
figures/fig_axes_panels.(png|pdf)
  a  Architecture benchmark   (encoder-decoder; 4 backbones swapped)
  b  Pretraining source       (only encoder init changes: random/ImageNet/CT)
  c  DixonFuse input design    (5 contrasts -> contrast-attention -> enc-dec)
  d  Multi-task learning       (1 shared encoder -> 2 task decoders)
Consistent 'hourglass' encoder-decoder glyph across panels so they are comparable.
"""
import os, glob, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib import colors as mcolors
import config as C

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
ENC = "#bcd6f2"; ENC_E = "#2b6cb0"
DEC = "#c4e5c8"; DEC_E = "#2f855a"
BOT = "#e2d3f7"; BOT_E = "#6b46c1"
FUSE = "#fbe6b6"; FUSE_E = "#b7791f"
INK = "#1a202c"; GREY = "#4a5568"; LINE = "#2d3748"


def muscle_cmap(n, seed=7):
    rng = np.random.RandomState(seed); c = rng.uniform(0.25, 1, (n, 3)); c[0] = 0
    return mcolors.ListedColormap(c)


def ct_overlay():
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(C.CACHE, "aattct_index.csv"))) if r["split"] == "test"]
    best = None
    for r in rows[::7][:40]:
        if (np.array(Image.open(r["vat"]).convert("L")) > 127).sum() > 6000:
            best = r; break
    r = best or rows[len(rows) // 2]
    g = np.array(Image.open(r["image"]).convert("L")).astype(float) / 255
    sat = np.array(Image.open(r["sat"]).convert("L")) > 127
    vat = np.array(Image.open(r["vat"]).convert("L")) > 127
    rgb = np.stack([g, g, g], -1); rgb[sat] = [0.16, 0.55, 1.0]; rgb[vat] = [1.0, 0.32, 0.24]
    return np.clip(rgb, 0, 1)


def load_imgs():
    meta = json.load(open(os.path.join(C.CACHE, "thigh5_meta.json")))
    f = sorted(glob.glob(os.path.join(C.CACHE, "thigh_slices5", f"{meta['subjects_test'][0]}_*.npz")))
    d = np.load(f[len(f) // 2]); water = d["img"][0]; ff = d["img"][4]; mask = d["mask"]
    w = (water - water.min()) / (np.ptp(water) + 1e-6)
    nc = int(mask.max()) + 1; over = muscle_cmap(max(nc, 14))(mask / max(nc - 1, 1))[..., :3]
    base = np.stack([w, w, w], -1); m = mask > 0
    seg = base.copy(); seg[m] = 0.4 * base[m] + 0.6 * over[m]
    return np.clip(w, 0, 1), np.clip(ff, 0, 1), np.clip(seg, 0, 1)


def hourglass(ax, x0, yc, w=2.4, H=1.5, hmin=0.55, enc_fc=ENC, enc_ec=ENC_E,
              dec_fc=DEC, dec_ec=DEC_E, enc_label="Encoder", dec_label="Decoder"):
    """Draw encoder trapezoid (narrowing) + bottleneck + decoder trapezoid (widening)."""
    ew = w * 0.42; dw = w * 0.42; bw = w * 0.16
    x1 = x0 + ew; x2 = x1 + bw; x3 = x2 + dw
    enc = Polygon([(x0, yc + H / 2), (x0, yc - H / 2), (x1, yc - hmin / 2), (x1, yc + hmin / 2)],
                  closed=True, facecolor=enc_fc, edgecolor=enc_ec, linewidth=1.4, zorder=3)
    bot = Rectangle((x1, yc - hmin / 2), bw, hmin, facecolor=BOT, edgecolor=BOT_E, linewidth=1.2, zorder=3)
    dec = Polygon([(x2, yc - hmin / 2), (x2, yc + hmin / 2), (x3, yc + H / 2), (x3, yc - H / 2)],
                  closed=True, facecolor=dec_fc, edgecolor=dec_ec, linewidth=1.4, zorder=3)
    ax.add_patch(enc); ax.add_patch(bot); ax.add_patch(dec)
    if enc_label:
        ax.text((x0 + x1) / 2, yc, enc_label, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=enc_ec, zorder=5, rotation=90)
    if dec_label:
        ax.text((x2 + x3) / 2, yc, dec_label, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=dec_ec, zorder=5, rotation=90)
    return x0, x3, x1, x2  # left, right, enc_right, dec_left


def img_at(ax, cx, cy, s, img, cmap=None, ec="#718096", label=None):
    a = ax.inset_axes([cx - s / 2, cy - s / 2, s, s], transform=ax.transData)
    a.imshow(img, cmap=cmap); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_edgecolor(ec); sp.set_linewidth(1.2)
    if label: a.set_title(label, fontsize=7.8, color=INK, pad=2, fontweight="bold")


def arr(ax, p1, p2, color=LINE, lw=1.5, ms=12, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=color, shrinkA=2, shrinkB=2, zorder=6, linestyle=ls))


def chip(ax, x, y, w, h, t, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                                linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8, color=INK, zorder=6, fontweight="bold")


def title(ax, letter, txt):
    ax.text(0.02, 0.97, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    ax.text(0.11, 0.965, txt, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top", color=INK)


def main():
    w_img, ff_img, seg = load_imgs()
    ct_img = ct_overlay()
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 7.2))
    for a in axs.flat:
        a.set_xlim(0, 10); a.set_ylim(0, 5); a.axis("off")

    # ---------------- a: architecture ----------------
    ax = axs[0, 0]; title(ax, "a", "Factor 1 — segmentation architecture")
    img_at(ax, 1.1, 2.6, 1.3, w_img, cmap="gray", ec="#718096", label="MRI")
    L, R, er, dl = hourglass(ax, 2.6, 2.6, w=3.6, H=1.7, hmin=0.6)
    img_at(ax, 8.7, 2.6, 1.3, seg, ec=DEC_E, label="mask")
    arr(ax, (1.75, 2.6), (2.6, 2.6)); arr(ax, (R, 2.6), (8.05, 2.6))
    ax.text(5.0, 0.7, "backbone  ∈  { U-Net,  U-Net++,  DeepLabV3+,  SegFormer }",
            ha="center", fontsize=9, color=INK, style="italic", fontweight="bold")

    # ---------------- b: pretraining source ----------------
    ax = axs[0, 1]; title(ax, "b", "Factor 2 — encoder pretraining source")
    encx = 3.6
    L, R, er, dl = hourglass(ax, encx, 2.4, w=3.6, H=1.7, hmin=0.6, enc_label="Encoder", dec_label="Decoder")
    img_at(ax, 8.9, 2.4, 1.15, seg, ec=DEC_E, label="mask")
    arr(ax, (R, 2.4), (8.3, 2.4))
    # three init sources -> encoder left face (only the encoder changes)
    srcs = [("random", "#e2e8f0", "#718096", 4.15, 3.0),
            ("ImageNet", ENC, ENC_E, 3.05, 2.4),
            ("CT-pretrained", FUSE, FUSE_E, 1.95, 1.8)]
    for t, fc, ec, ysrc, ytgt in srcs:
        chip(ax, 0.5, ysrc - 0.32, 2.0, 0.64, t, fc, ec)
        arr(ax, (2.5, ysrc), (encx, ytgt), color=ec, lw=1.4, ms=11)
    ax.text(5.0, 0.55, "only the encoder initialisation changes", ha="center",
            fontsize=8.6, color=INK, style="italic", fontweight="bold")

    # ---------------- c: DixonFuse ----------------
    ax = axs[1, 0]; title(ax, "c", "Factor 3 — physics-guided input design")
    names = ["Water", "Fat", "In", "Opp", "FF"]
    for i, nm in enumerate(names):
        y = 3.7 - i * 0.42
        fc = FUSE if nm == "FF" else "#eef2f7"
        ax.add_patch(Rectangle((0.5, y), 1.15, 0.36, facecolor=fc, edgecolor=FUSE_E if nm == "FF" else "#94a3b8", linewidth=1.0, zorder=4))
        ax.text(1.07, y + 0.18, nm, ha="center", va="center", fontsize=7, color=INK, zorder=5, fontweight="bold")
    ax.text(1.07, 3.95, "5 contrasts", ha="center", fontsize=7.3, color=INK, fontweight="bold")
    chip(ax, 2.15, 2.3, 1.5, 0.9, "contrast\nattention\n(SE gate)", FUSE, FUSE_E)
    arr(ax, (1.65, 2.75), (2.15, 2.75), lw=1.3, ms=10)
    L, R, er, dl = hourglass(ax, 3.95, 2.55, w=3.3, H=1.6, hmin=0.55, enc_label="Encoder", dec_label="Decoder")
    arr(ax, (3.65, 2.75), (3.95 + 0.05, 2.6), lw=1.3, ms=10)
    img_at(ax, 8.9, 2.55, 1.1, seg, ec=DEC_E, label="mask")
    arr(ax, (R, 2.55), (8.35, 2.55))
    # per-contrast squeeze-and-excitation gate, drawn as an equation
    ax.text(5.0, 1.12, r"$s=\sigma\!\left(W_2\,\delta(W_1\,\mathrm{GAP}(x))\right),\quad \hat{x}=s\odot x$",
            ha="center", fontsize=10, color=INK)
    ax.text(5.0, 0.55, "per-image gate  s ∈ [0,1] per contrast reweights each channel;  "
            "FF = Fat/(Fat+Water)", ha="center", fontsize=7.8, color=INK, style="italic", fontweight="bold")

    # ---------------- d: multi-task ----------------
    ax = axs[1, 1]; title(ax, "d", "Factor 4 — multi-task (shared encoder)")
    # shared encoder trapezoid in middle
    ex0, ex1, ey = 3.0, 4.7, 2.5
    enc = Polygon([(ex0, ey + 0.95), (ex0, ey - 0.95), (ex1, ey - 0.32), (ex1, ey + 0.32)],
                  closed=True, facecolor=ENC, edgecolor=ENC_E, linewidth=1.5, zorder=3)
    ax.add_patch(enc)
    ax.text((ex0 + ex1) / 2, ey, "shared\nencoder", ha="center", va="center", fontsize=8, fontweight="bold", color=ENC_E, zorder=5)
    # two inputs (real CT with SAT/VAT overlay + real MRI water)
    img_at(ax, 1.0, 3.5, 1.0, ct_img, ec=ENC_E, label="CT (SAT/VAT)")
    img_at(ax, 1.0, 1.5, 1.0, w_img, cmap="gray", ec=ENC_E, label="MRI")
    arr(ax, (1.5, 3.4), (ex0, ey + 0.5), lw=1.3, ms=10)
    arr(ax, (1.5, 1.6), (ex0, ey - 0.5), lw=1.3, ms=10)
    # two decoders
    for y, fc, ec, lab, mfc in [(3.55, DEC, DEC_E, "CT decoder", "SAT/VAT"), (1.45, DEC, DEC_E, "MRI decoder", "muscles")]:
        dx0, dx1 = 5.4, 7.0
        dec = Polygon([(dx0, y - 0.28), (dx0, y + 0.28), (dx1, y + 0.7), (dx1, y - 0.7)],
                      closed=True, facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=3)
        ax.add_patch(dec)
        ax.text((dx0 + dx1) / 2, y, lab, ha="center", va="center", fontsize=7.2, fontweight="bold", color=ec, rotation=90, zorder=5)
        arr(ax, (ex1, ey + (0.35 if y > 2.5 else -0.35)), (dx0, y), lw=1.3, ms=10)
        chip(ax, 7.4, y - 0.28, 1.7, 0.56, mfc, "#eef2f7", "#94a3b8")
        arr(ax, (dx1, y), (7.4, y), lw=1.2, ms=9)
    ax.text(5.0, 0.55, "tied encoder weights; robust to normalisation scheme", ha="center", fontsize=8.2, color=INK, style="italic", fontweight="bold")

    plt.tight_layout(pad=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG, f"fig_axes_panels.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote fig_axes_panels.(png|pdf)")


if __name__ == "__main__":
    main()
