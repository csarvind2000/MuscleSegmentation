"""Shared-encoder multi-task segmentation across two body-composition modalities.

Design:
  input adapter (1x1 conv, task-specific) : CT 1->3ch , MRI 4->3ch
  shared encoder (resnet34, ImageNet)     : identical weights for both tasks
  task-specific Unet decoder + seg head    : CT 3-class , MRI 14-class

Trained by alternating one CT batch and one MRI batch per step (losses summed).
Compared against the single-task Unet-r34 baselines to test whether a shared
cross-modality representation helps the small (25-subject) MRI task.
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import config as C
from datasets import AATTCTDataset, ThighSliceDataset
from train import DiceCELoss, evaluate


class Branch(nn.Module):
    """1x1 adapter -> per-task UNet (own encoder BN, own decoder/head).

    The encoder CONV weights are tied across branches (shared representation),
    but each branch keeps its OWN BatchNorm layers -> domain-specific BN, which
    fixes the cross-modality BN-statistics collapse.
    """
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.adapter = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                            in_channels=3, classes=num_classes)

    def forward(self, x):
        return self.net(self.adapter(x))


def tie_conv_weights(src, dst):
    """Share every Conv2d/Linear weight (and bias) from src.encoder into dst.encoder.
    BatchNorm layers are left independent (domain-specific)."""
    def leaves(m, types):
        return [x for x in m.modules() if isinstance(x, types)]
    s = leaves(src, (nn.Conv2d, nn.Linear))
    d = leaves(dst, (nn.Conv2d, nn.Linear))
    assert len(s) == len(d), (len(s), len(d))
    n = 0
    for a, b in zip(s, d):
        if a.weight.shape == b.weight.shape:
            b.weight = a.weight
            if a.bias is not None and b.bias is not None:
                b.bias = a.bias
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"

    idx = os.path.join(C.CACHE, "aattct_index.csv")
    a_tr = DataLoader(AATTCTDataset(idx, "train", augment=True), args.bs, True, num_workers=args.workers, pin_memory=True, drop_last=True)
    a_va = DataLoader(AATTCTDataset(idx, "val"), args.bs, False, num_workers=args.workers, pin_memory=True)
    a_te = DataLoader(AATTCTDataset(idx, "test"), args.bs, False, num_workers=args.workers, pin_memory=True)
    tmeta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    tsplit = json.load(open(os.path.join(C.CACHE, "thigh_split.json")))
    t_tr = DataLoader(ThighSliceDataset(tsplit["train"], augment=True), args.bs, True, num_workers=args.workers, pin_memory=True, drop_last=True)
    t_va = DataLoader(ThighSliceDataset(tsplit["val"]), args.bs, False, num_workers=args.workers, pin_memory=True)
    t_te = DataLoader(ThighSliceDataset(tsplit["test"]), args.bs, False, num_workers=args.workers, pin_memory=True)

    a_nc = C.AATTCT_NUM_CLASSES
    t_nc = tmeta["num_classes"]

    ct = Branch(1, a_nc).to(dev)
    mri = Branch(tmeta["in_channels"], t_nc).to(dev)
    n_tied = tie_conv_weights(ct.net.encoder, mri.net.encoder)
    print(f"[MTL] tied {n_tied} encoder conv/linear weight tensors (domain-specific BN kept separate)", flush=True)

    params = list(dict.fromkeys(list(ct.parameters()) + list(mri.parameters())))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit_a = DiceCELoss(a_nc).to(dev); crit_t = DiceCELoss(t_nc).to(dev)
    scaler = torch.cuda.amp.GradScaler()

    best = -1; bp_a = os.path.join(C.CKPT, "mtl_ct.pt"); bp_m = os.path.join(C.CKPT, "mtl_mri.pt")
    t0 = time.time()
    for ep in range(args.epochs):
        ct.train(); mri.train()
        ai = iter(a_tr); ti = iter(t_tr)
        steps = max(len(a_tr), len(t_tr)); run = 0.0
        for _ in range(steps):
            try: aimg, amask = next(ai)
            except StopIteration: ai = iter(a_tr); aimg, amask = next(ai)
            try: timg, tmask = next(ti)
            except StopIteration: ti = iter(t_tr); timg, tmask = next(ti)
            aimg, amask = aimg.to(dev), amask.to(dev)
            timg, tmask = timg.to(dev), tmask.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit_a(ct(aimg), amask) + crit_t(mri(timg), tmask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        av, _ = evaluate(ct, a_va, a_nc, dev)
        mv, _ = evaluate(mri, t_va, t_nc, dev)
        score = float(np.nanmean(av[1:]) + np.nanmean(mv[1:]))
        print(f"[MTL] ep{ep+1}/{args.epochs} loss={run/steps:.4f} ct_val={np.nanmean(av[1:]):.4f} mri_val={np.nanmean(mv[1:]):.4f}", flush=True)
        if score > best:
            best = score
            torch.save(ct.state_dict(), bp_a); torch.save(mri.state_dict(), bp_m)

    ct.load_state_dict(torch.load(bp_a)); mri.load_state_dict(torch.load(bp_m))
    ad, ai_ = evaluate(ct, a_te, a_nc, dev)
    md, mi_ = evaluate(mri, t_te, t_nc, dev)
    res = {"model": "mtl_shared_resnet34", "epochs": args.epochs,
           "aattct": {"classes": C.AATTCT_CLASSES,
                      "test_dice_per_class": [None if np.isnan(x) else round(float(x),4) for x in ad],
                      "test_dice_fg_mean": round(float(np.nanmean(ad[1:])),4)},
           "thigh": {"classes": tmeta["classes"],
                     "test_dice_per_class": [None if np.isnan(x) else round(float(x),4) for x in md],
                     "test_dice_fg_mean": round(float(np.nanmean(md[1:])),4)},
           "minutes": round((time.time()-t0)/60,1)}
    _out = "mtl_shared.json" if args.seed == C.SEED else f"mtl_shared_s{args.seed}.json"
    json.dump(res, open(os.path.join(C.RESULTS, _out), "w"), indent=2)
    print("[MTL] DONE ct", res["aattct"]["test_dice_fg_mean"], "mri", res["thigh"]["test_dice_fg_mean"], flush=True)


if __name__ == "__main__":
    main()
