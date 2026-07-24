"""Domain-randomisation retraining for TRUE zero-shot external transfer (Tier 1).

Retrains the 4-channel primary thigh model with heavy APPEARANCE augmentation designed to
span the external cohort's look (nonlinear contrast, per-channel gain, smooth bias field,
Gaussian blur to mimic thick-slice / resolution loss, and noise). No external data of any
kind is used in training, so the external evaluation remains genuinely zero-shot. The
checkpoint is selected on the in-domain validation Dice (never on external), then applied
directly to all 66 HuashanMyo subjects.

Writes results/ext_dr.json and checkpoints/thigh_unet_r34_dr.pt
"""
import os, glob, json, time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom
from torch.utils.data import Dataset, DataLoader
import config as C
from models import build_model
from datasets import _aug2d_multi
import zeroshot_external as Z

EPOCHS = 80


def dr_appearance(img):
    """Heavy appearance randomisation on a 4xHxW, per-volume z-scored slice.
    All transforms keep the scale ~O(1) so inputs stay compatible with z-scored test data."""
    out = img.copy()
    Cc, Hh, Ww = out.shape
    # 1. per-channel signed gamma (nonlinear contrast) + gain/bias
    for c in range(Cc):
        if np.random.rand() < 0.8:
            g = np.random.uniform(0.7, 1.5)
            out[c] = np.sign(out[c]) * (np.abs(out[c]) ** g)
        out[c] = np.random.uniform(0.8, 1.2) * out[c] + np.random.uniform(-0.2, 0.2)
    # 2. smooth multiplicative bias field (shared across channels)
    if np.random.rand() < 0.7:
        coarse = np.random.randn(6, 6).astype(np.float32)
        field = zoom(coarse, (Hh / 6.0, Ww / 6.0), order=3)[:Hh, :Ww]
        field = 1.0 + 0.30 * (field / (np.abs(field).max() + 1e-6))
        out = out * field[None]
    # 3. Gaussian blur (resolution loss / partial-volume look)
    if np.random.rand() < 0.6:
        s = np.random.uniform(0.4, 2.2)
        for c in range(Cc):
            out[c] = gaussian_filter(out[c], s)
    # 4. Gaussian noise
    if np.random.rand() < 0.5:
        out = out + np.random.randn(*out.shape).astype(np.float32) * np.random.uniform(0.01, 0.08)
    return out.astype(np.float32)


class DRThigh(Dataset):
    def __init__(self, files, augment=False):
        self.files, self.augment = files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = d["img"].astype(np.float32)          # [4,H,W]
        mask = d["mask"].astype(np.int64)
        if self.augment:
            img, mask = _aug2d_multi(img, mask)    # geometric
            img = dr_appearance(img)               # appearance domain randomisation
        return torch.from_numpy(np.ascontiguousarray(img)).float(), torch.from_numpy(mask).long()


def dicece(logits, target, nc):
    ce = torch.nn.functional.cross_entropy(logits, target)
    probs = torch.softmax(logits, 1)
    t1h = torch.nn.functional.one_hot(target, nc).permute(0, 3, 1, 2).float()
    inter = (probs * t1h).sum((0, 2, 3)); denom = probs.sum((0, 2, 3)) + t1h.sum((0, 2, 3))
    return ce + (1 - ((2 * inter + 1) / (denom + 1)).mean())


def train_dr(model="unet_r34", epochs=EPOCHS):
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    split = json.load(open(os.path.join(C.CACHE, "thigh_split.json")))
    nc = meta["num_classes"]; dev = "cuda"
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    tr = DataLoader(DRThigh(split["train"], True), batch_size=8, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=True)
    va = DataLoader(DRThigh(split["val"]), batch_size=8, num_workers=4)
    net = build_model(model, 4, nc).to(dev)                 # 4-ch, ImageNet init
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler()
    ckpt = os.path.join(C.CKPT, f"thigh_{model}_dr.pt"); best = -1; t0 = time.time()
    for ep in range(epochs):
        net.train()
        for img, mask in tr:
            img, mask = img.to(dev), mask.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = dicece(net(img), mask, nc)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        net.eval(); ds = np.zeros(nc); dc = np.zeros(nc)
        with torch.no_grad():
            for img, mask in va:
                pr = net(img.to(dev)).argmax(1).cpu().numpy(); m = mask.numpy()
                for c in range(1, nc):
                    p = (pr == c); t = (m == c); dn = p.sum() + t.sum()
                    if dn > 0: ds[c] += 2 * (p & t).sum() / dn; dc[c] += 1
        vfg = float(np.nanmean(np.where(dc > 0, ds / np.maximum(dc, 1), np.nan)[1:]))
        if vfg > best: best = vfg; torch.save(net.state_dict(), ckpt)
        print(f"[dr-train] ep{ep+1}/{epochs} val_fg={vfg:.4f} best={best:.4f}", flush=True)
    print(f"[dr-train] done best_val_fg={best:.4f} in {(time.time()-t0)/60:.1f} min", flush=True)
    return ckpt, round(best, 4)


def zeroshot_dr(ckpt, model="unet_r34"):
    dev = "cuda"
    net = build_model(model, 4, 14, encoder_weights=None)
    net.load_state_dict(torch.load(ckpt, map_location="cpu")); net = net.to(dev).eval()
    sids = sorted({os.path.basename(f).replace(".nii.gz", "")
                   for f in glob.glob(os.path.join(Z.EXT, "Label", "*.nii.gz"))})
    per_subj = []
    for sid in sids:
        s = Z.subject_entries(sid)
        if not s or not s["entries"]:
            continue
        slices = [(e["img"], e["mask"]) for e in s["entries"]]
        pred = Z.predict_subject(net, slices, dev)
        d = Z.subject_dice(pred, [e["mask"] for e in s["entries"]])
        if d:
            per_subj.append(float(np.mean(list(d.values()))))
    a = np.array(per_subj)
    print(f"[dr-zeroshot] n={len(a)} mean={a.mean():.4f} sd={a.std(ddof=1):.4f} "
          f"range={a.min():.3f}-{a.max():.3f}", flush=True)
    return a


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["unet_r34"])
    args = ap.parse_args()
    for model in args.models:
        print(f"\n===== domain-randomisation training: {model} =====", flush=True)
        ck, bestval = train_dr(model, EPOCHS)
        a = zeroshot_dr(ck, model)
        outf = "ext_dr.json" if model == "unet_r34" else f"ext_dr_{model}.json"
        json.dump({"method": "domain-randomisation retraining (true zero-shot; no external data used)",
                   "model": f"{model} (4-channel), ImageNet init, appearance domain randomisation",
                   "primary_val_fg_dice": bestval,
                   "n": len(a), "mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4),
                   "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
                   "per_subject": [round(float(x), 4) for x in a]},
                  open(os.path.join(C.RESULTS, outf), "w"), indent=2)
        print(f"wrote results/{outf}", flush=True)
