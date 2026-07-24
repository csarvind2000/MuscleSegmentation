"""Per-muscle Dice on the external cohort for the headline config (ImageNet init,
Water+Fat+FF, full n=46). Retrains the config (checkpoints are not persisted per-run)
for seeds 0-2 and writes the per-class (per-muscle) Dice averaged over test subjects and
seeds, so we can draw the external per-muscle spider. Numbers match the 0.865 reported.

Writes results/ext_permuscle_wff_imagenet_n46.json
"""
import os, json, glob, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader
import config as C
from datasets import ThighSliceDataset
from train import DiceCELoss, evaluate
from train_external import build, Sel, sfiles

DEV = "cuda"


@torch.no_grad()
def per_class_dice(model, subjects, cache, nc):
    """Mean per-class Dice over subjects (volumetric: pool slices within subject)."""
    per_sub = []
    for s in subjects:
        dl = DataLoader(ThighSliceDataset(sorted(glob.glob(os.path.join(cache, f"{s}_*.npz")))),
                        8, False, num_workers=4)
        inter = np.zeros(nc); den = np.zeros(nc)
        for img, mask in dl:
            pred = model(img.to(DEV)).argmax(1).cpu()
            for c in range(nc):
                p = (pred == c); t = (mask == c)
                inter[c] += (p & t).sum().item(); den[c] += p.sum().item() + t.sum().item()
        per_sub.append(np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan))
    return np.nanmean(np.stack(per_sub), 0)  # nc-vector


def train_one(init, channels, n, seed, epochs=70, bs=8):
    cache = os.path.join(C.CACHE, "ext_slices3")
    meta = json.load(open(os.path.join(C.CACHE, "ext_meta.json")))
    nc = meta["num_classes"]; in_ch = 2 if channels == "wf" else 3
    pool, val, test = meta["subjects_train"], meta["subjects_val"], meta["subjects_test"]
    random.seed(seed)
    chosen = sorted(random.sample(pool, n)) if n and n < len(pool) else pool
    torch.manual_seed(seed); np.random.seed(seed)
    tr = DataLoader(ThighSliceDataset(sfiles(chosen, cache), augment=True), bs, True,
                    num_workers=6, pin_memory=True, drop_last=len(chosen) > 1)
    va = DataLoader(ThighSliceDataset(sfiles(val, cache)), bs, False, num_workers=6, pin_memory=True)
    model = Sel(build(init, in_ch, nc), in_ch).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = DiceCELoss(nc).to(DEV); scaler = torch.cuda.amp.GradScaler()
    best = -1; tmp = os.path.join(C.CKPT, f"_extpm_{seed}.pt")
    for ep in range(epochs):
        model.train()
        for img, mask in tr:
            img, mask = img.to(DEV), mask.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        vd, _ = evaluate(model, va, nc, DEV); vfg = float(np.nanmean(vd[1:]))
        if vfg > best:
            best = vfg; torch.save(model.state_dict(), tmp)
    model.load_state_dict(torch.load(tmp)); model.eval()
    pc = per_class_dice(model, test, cache, nc)
    os.remove(tmp)
    return pc, nc


def main():
    t0 = time.time()
    init, channels, n = "imagenet", "wff", 46
    seeds = [0, 1, 2]
    mats = []
    for sd in seeds:
        pc, nc = train_one(init, channels, n, sd)
        mats.append(pc)
        print(f"seed {sd}: fg-mean={np.nanmean(pc[1:]):.4f}", flush=True)
    M = np.stack(mats)                      # seeds x nc
    mean_pc = np.nanmean(M, 0)              # nc
    sd_pc = np.nanstd(M, 0)                 # nc across seeds
    out = {"init": init, "channels": channels, "n": n, "seeds": seeds,
           "num_classes": int(nc),
           "per_class_dice_mean": [round(float(x), 4) for x in mean_pc],
           "per_class_dice_sd": [round(float(x), 4) for x in sd_pc],
           "fg_mean": round(float(np.nanmean(mean_pc[1:])), 4),
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(os.path.join(C.RESULTS, "ext_permuscle_wff_imagenet_n46.json"), "w"), indent=2)
    print("fg-mean over seeds:", out["fg_mean"], "| per-muscle:",
          [round(x, 3) for x in mean_pc[1:]], flush=True)


if __name__ == "__main__":
    main()
