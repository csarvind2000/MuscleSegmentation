"""Multi-task normalisation study (reviewer item 8).

Trains a shared-encoder CT+MRI model under different normalisation schemes to show
that the CT-branch collapse under shared BatchNorm is a normalisation issue, and to
compare alternatives:
  shared_bn    : one shared encoder with standard BatchNorm  (reproduces collapse)
  groupnorm    : one shared encoder, BN replaced by GroupNorm
  instancenorm : one shared encoder, BN replaced by InstanceNorm
(domain-specific BN is trained separately by train_multitask.py.)

Usage: python train_mtl_norm.py --norm shared_bn --seed 0 --epochs 60
Writes results/mtlnorm_<norm>_s<seed>.json
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import config as C
from datasets import AATTCTDataset, ThighSliceDataset
from train import DiceCELoss, evaluate


def convert_norm(module, kind):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            if kind == "groupnorm":
                ng = 16 if c % 16 == 0 else (8 if c % 8 == 0 else 1)
                setattr(module, name, nn.GroupNorm(ng, c))
            elif kind == "instancenorm":
                setattr(module, name, nn.InstanceNorm2d(c, affine=True, track_running_stats=False))
        else:
            convert_norm(child, kind)


class SharedMTL(nn.Module):
    def __init__(self, norm, a_nc, t_nc, mri_in):
        super().__init__()
        base = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=a_nc)
        self.encoder = base.encoder
        if norm in ("groupnorm", "instancenorm"):
            convert_norm(self.encoder, norm)
        self.ct_adapter = nn.Conv2d(1, 3, 1)
        self.mri_adapter = nn.Conv2d(mri_in, 3, 1)
        self.ct_dec = base.decoder
        self.ct_head = base.segmentation_head
        mri = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=t_nc)
        self.mri_dec = mri.decoder
        self.mri_head = mri.segmentation_head

    def ct(self, x):
        return self.ct_head(self.ct_dec(self.encoder(self.ct_adapter(x))))

    def mri(self, x):
        return self.mri_head(self.mri_dec(self.encoder(self.mri_adapter(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", required=True, choices=["shared_bn", "groupnorm", "instancenorm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"

    idx = os.path.join(C.CACHE, "aattct_index.csv")
    a_tr = DataLoader(AATTCTDataset(idx, "train", augment=True), args.bs, True, num_workers=args.workers, pin_memory=True, drop_last=True)
    a_te = DataLoader(AATTCTDataset(idx, "test"), args.bs, False, num_workers=args.workers, pin_memory=True)
    tmeta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    tsplit = json.load(open(os.path.join(C.CACHE, "thigh_split.json")))
    t_tr = DataLoader(ThighSliceDataset(tsplit["train"], augment=True), args.bs, True, num_workers=args.workers, pin_memory=True, drop_last=True)
    t_te = DataLoader(ThighSliceDataset(tsplit["test"]), args.bs, False, num_workers=args.workers, pin_memory=True)
    a_nc = C.AATTCT_NUM_CLASSES; t_nc = tmeta["num_classes"]

    model = SharedMTL(args.norm, a_nc, t_nc, tmeta["in_channels"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit_a = DiceCELoss(a_nc).to(dev); crit_t = DiceCELoss(t_nc).to(dev)
    scaler = torch.cuda.amp.GradScaler()
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); ai = iter(a_tr); ti = iter(t_tr)
        for _ in range(max(len(a_tr), len(t_tr))):
            try: aimg, amask = next(ai)
            except StopIteration: ai = iter(a_tr); aimg, amask = next(ai)
            try: timg, tmask = next(ti)
            except StopIteration: ti = iter(t_tr); timg, tmask = next(ti)
            aimg, amask, timg, tmask = aimg.to(dev), amask.to(dev), timg.to(dev), tmask.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit_a(model.ct(aimg), amask) + crit_t(model.mri(timg), tmask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()

    # eval with two thin wrappers so evaluate() can call model(img)
    ct_w = type("W", (nn.Module,), {"forward": lambda s, x: model.ct(x)})().to(dev)
    mri_w = type("W", (nn.Module,), {"forward": lambda s, x: model.mri(x)})().to(dev)
    ad, _ = evaluate(ct_w, a_te, a_nc, dev)
    md, _ = evaluate(mri_w, t_te, t_nc, dev)
    res = {"norm": args.norm, "seed": args.seed,
           "ct_dice_fg": round(float(np.nanmean(ad[1:])), 4),
           "mri_dice_fg": round(float(np.nanmean(md[1:])), 4),
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(os.path.join(C.RESULTS, f"mtlnorm_{args.norm}_s{args.seed}.json"), "w"), indent=2)
    print(f"[mtlnorm_{args.norm}_s{args.seed}] ct={res['ct_dice_fg']} mri={res['mri_dice_fg']}", flush=True)


if __name__ == "__main__":
    main()
