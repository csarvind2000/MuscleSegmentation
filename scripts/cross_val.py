"""Repeated k-fold cross-validation with per-subject metrics (reviewer item 2).

Uses a fixed 5-fold partition of the 25 MRI subjects (every subject tested once).
For a given fold, trains on a subsample of the remaining subjects and evaluates Dice
PER TEST SUBJECT, so that downstream analysis can compute 95% CIs, TOST equivalence,
and Holm-corrected comparisons at the subject level.

--model init:random|init:imagenet|init:ct  (4-ch Unet, varying encoder init)
--model method:dixon4|method:dixonfuse       (DixonFuse variants, 5-ch cache)
--region thigh|calf   --n <train subjects>   --fold 0..4   --repeat 0..2
Writes results/cv_<tag>.json  with {subject: dice} for the test fold.
"""
import os, json, glob, argparse, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import config as C
from datasets import ThighSliceDataset
from models_method import DixonFuseUNet
from train import DiceCELoss, dice_score
from train_lowdata import adapt_ct_encoder


def all_subjects():
    return [f"{i:02d}" for i in range(1, 26)]


def folds(repeat, k=5):
    subs = all_subjects()
    rng = random.Random(1000 + repeat)
    rng.shuffle(subs)
    return [subs[i::k] for i in range(k)]  # k test-folds


def sfiles(subjects, cache):
    return [f for s in subjects for f in sorted(glob.glob(os.path.join(cache, f"{s}_*.npz")))]


class Sel4(nn.Module):
    """Wrap a 4-ch Unet so it reads the first 4 channels of the 5-ch cache."""
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x): return self.net(x[:, :4])


def build(model_spec, nc):
    kind, name = model_spec.split(":")
    if kind == "method":
        return DixonFuseUNet(name, nc)
    ew = "imagenet" if name in ("imagenet", "ct") else None
    net = smp.Unet("resnet34", encoder_weights=ew, in_channels=4, classes=nc)
    if name == "ct":
        adapt_ct_encoder(net, os.path.join(C.CKPT, "aattct_unet_r34.pt"))
    return Sel4(net)


@torch.no_grad()
def per_subject_dice(model, subjects, cache, nc, dev, aux=False):
    out = {}
    for s in subjects:
        files = sorted(glob.glob(os.path.join(cache, f"{s}_*.npz")))
        dl = DataLoader(ThighSliceDataset(files), 8, False, num_workers=4, pin_memory=True)
        inter = np.zeros(nc); denom = np.zeros(nc)
        for img, mask in dl:
            logits = model(img.to(dev))
            if aux:                       # ffsup: first nc channels are the segmentation
                logits = logits[:, :nc]
            pred = logits.argmax(1).cpu()
            for c in range(nc):
                p = (pred == c); t = (mask == c)
                inter[c] += (p & t).sum().item(); denom[c] += p.sum().item() + t.sum().item()
        d = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), np.nan)
        out[s] = round(float(np.nanmean(d[1:])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--region", default="thigh", choices=["thigh", "calf"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ff_weight", type=float, default=1.0, help="weight of the FF auxiliary loss (ffsup only)")
    args = ap.parse_args()

    cache = os.path.join(C.CACHE, f"{args.region}_slices5")
    meta = json.load(open(os.path.join(C.CACHE, f"{args.region}5_meta.json")))
    nc = meta["num_classes"]
    fs = folds(args.repeat)
    test = fs[args.fold]
    pool = [s for s in all_subjects() if s not in test]
    seed = 100 * args.repeat + args.fold
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    train = sorted(random.sample(pool, min(args.n, len(pool))))
    dev = "cuda"

    tr = DataLoader(ThighSliceDataset(sfiles(train, cache), augment=True), args.bs, True,
                    num_workers=args.workers, pin_memory=True, drop_last=len(train) > 1)
    is_ffsup = args.model.endswith(":ffsup")   # FF as physics supervision (Option A)
    model = build(args.model, nc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = DiceCELoss(nc).to(dev); mse = nn.MSELoss(); scaler = torch.cuda.amp.GradScaler()
    for ep in range(args.epochs):
        model.train()
        for img, mask in tr:
            img, mask = img.to(dev), mask.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(img)
                if is_ffsup:
                    seg = out[:, :nc]
                    ff_pred = torch.sigmoid(out[:, nc:nc + 1])
                    ff_tgt = img[:, 4:5]                    # measured fat-fraction (physics target)
                    loss = crit(seg, mask) + args.ff_weight * mse(ff_pred, ff_tgt)
                else:
                    loss = crit(out, mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    model.eval()
    ps = per_subject_dice(model, test, cache, nc, dev, aux=is_ffsup)
    tag = f"cv_{args.region}_{args.model.replace(':','-')}_n{args.n}_f{args.fold}_r{args.repeat}"
    if is_ffsup and args.ff_weight != 1.0:
        tag += f"_w{args.ff_weight:g}"
    json.dump({"model": args.model, "region": args.region, "n": args.n, "fold": args.fold,
               "repeat": args.repeat, "per_subject_dice": ps}, open(os.path.join(C.RESULTS, tag + ".json"), "w"), indent=2)
    print(f"[{tag}] " + " ".join(f"{k}:{v}" for k, v in ps.items()), flush=True)


if __name__ == "__main__":
    main()
