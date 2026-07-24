"""Cross-modality transfer + data-efficiency study on thigh-muscle MRI.

Trains U-Net(ResNet-34) on N MRI training subjects under three encoder inits:
  random   : encoder_weights=None
  imagenet : encoder_weights='imagenet'
  ct       : encoder pretrained on AATTCT adipose task (checkpoints/aattct_unet_r34.pt)

The CT encoder is 1-channel; the MRI model is 4-channel. We transfer all encoder
weights and adapt the stem conv (1->4 ch) by tiling/scaling. Test set is the fixed
5-subject held-out split; only the number of *training* subjects varies.

Usage:
  python train_lowdata.py --n 4 --init ct --seed 0 --epochs 60
Writes results/lowdata_<init>_n<N>_s<seed>.json
"""
import os, json, glob, argparse, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import config as C
from datasets import ThighSliceDataset
from train import DiceCELoss, evaluate

def subject_files(subjects, cache_dir):
    out = []
    for s in subjects:
        out += sorted(glob.glob(os.path.join(cache_dir, f"{s}_*.npz")))
    return out


def adapt_ct_encoder(model, ct_ckpt):
    """Load CT U-Net encoder weights into a 4-ch MRI U-Net, adapting the stem conv."""
    sd = torch.load(ct_ckpt, map_location="cpu")
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    tgt = model.encoder.state_dict()
    for k, v in enc.items():
        if k not in tgt:
            continue
        if v.shape == tgt[k].shape:
            tgt[k] = v
        elif v.dim() == 4 and v.shape[1] == 1 and tgt[k].shape[1] == 4:
            # stem conv 1ch -> 4ch: tile & scale to preserve activation magnitude
            tgt[k] = v.repeat(1, 4, 1, 1) / 4.0
    model.encoder.load_state_dict(tgt)
    return model


def build(init, in_ch, nc, region="thigh"):
    if init == "random":
        m = smp.Unet("resnet34", encoder_weights=None, in_channels=in_ch, classes=nc)
    elif init == "imagenet":
        m = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=nc)
    elif init == "ct":
        m = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=nc)
        m = adapt_ct_encoder(m, os.path.join(C.CKPT, "aattct_unet_r34.pt"))
    elif init == "ssl":
        m = smp.Unet("resnet34", encoder_weights=None, in_channels=in_ch, classes=nc)
        sd = torch.load(os.path.join(C.CKPT, f"ssl_encoder_{region}.pt"), map_location="cpu")
        m.encoder.load_state_dict(sd)
    else:
        raise ValueError(init)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--init", required=True, choices=["random", "imagenet", "ct", "ssl"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--region", default="thigh", choices=["thigh", "calf"])
    args = ap.parse_args()

    cache_dir = os.path.join(C.CACHE, f"{args.region}_slices")
    meta = json.load(open(os.path.join(C.CACHE, f"{args.region}_meta.json")))
    nc, in_ch = meta["num_classes"], meta["in_channels"]
    train_pool = meta["subjects_train"]
    val_subj = meta["subjects_val"]
    test_subj = meta["subjects_test"]

    random.seed(args.seed)
    chosen = sorted(random.sample(train_pool, args.n))

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    tr = DataLoader(ThighSliceDataset(subject_files(chosen, cache_dir), augment=True), args.bs, True,
                    num_workers=args.workers, pin_memory=True, drop_last=len(chosen) > 1)
    va = DataLoader(ThighSliceDataset(subject_files(val_subj, cache_dir)), args.bs, False,
                    num_workers=args.workers, pin_memory=True)
    te = DataLoader(ThighSliceDataset(subject_files(test_subj, cache_dir)), args.bs, False,
                    num_workers=args.workers, pin_memory=True)

    model = build(args.init, in_ch, nc, args.region).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = DiceCELoss(nc).to(dev)
    scaler = torch.cuda.amp.GradScaler()

    rp = "" if args.region == "thigh" else f"{args.region}_"
    tag = f"lowdata_{rp}{args.init}_n{args.n}_s{args.seed}"
    best = -1; bpath = os.path.join(C.CKPT, tag + ".pt")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); run = 0.0
        for img, mask in tr:
            img, mask = img.to(dev), mask.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        vd, _ = evaluate(model, va, nc, dev)
        vfg = float(np.nanmean(vd[1:]))
        if vfg > best:
            best = vfg; torch.save(model.state_dict(), bpath)
    model.load_state_dict(torch.load(bpath))
    td, ti = evaluate(model, te, nc, dev)
    res = {"init": args.init, "n_subjects": args.n, "seed": args.seed,
           "subjects_used": chosen, "epochs": args.epochs,
           "test_dice_per_class": [None if np.isnan(x) else round(float(x), 4) for x in td],
           "test_dice_fg_mean": round(float(np.nanmean(td[1:])), 4),
           "test_iou_fg_mean": round(float(np.nanmean(ti[1:])), 4),
           "best_val_dice_fg": round(float(best), 4),
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(os.path.join(C.RESULTS, tag + ".json"), "w"), indent=2)
    print(f"[{tag}] DONE test_dice_fg={res['test_dice_fg_mean']} ({res['minutes']}min)", flush=True)
    if os.path.exists(bpath) and args.init != "imagenet":  # keep imagenet ckpts small footprint
        pass


if __name__ == "__main__":
    main()
