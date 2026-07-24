"""Train a DixonFuse variant on thigh-muscle MRI (5-channel cache).

Usage:
  python train_method.py --variant dixonfuse --n 18 --seed 0 --epochs 80
  python train_method.py --variant dixon4    --n 4  --seed 1 --epochs 80
Writes results/method_<variant>_n<N>_s<seed>.json
"""
import os, json, glob, argparse, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from datasets import ThighSliceDataset
from models_method import DixonFuseUNet
from train import DiceCELoss, evaluate


def subject_files(subjects, cache5):
    return [f for s in subjects for f in sorted(glob.glob(os.path.join(cache5, f"{s}_*.npz")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["wateronly", "dixon4", "gateonly", "dixon5", "dixonfuse"])
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--region", default="thigh", choices=["thigh", "calf"])
    args = ap.parse_args()

    cache5 = os.path.join(C.CACHE, f"{args.region}_slices5")
    meta = json.load(open(os.path.join(C.CACHE, f"{args.region}5_meta.json")))
    nc = meta["num_classes"]
    train_pool = meta["subjects_train"]; val_subj = meta["subjects_val"]; test_subj = meta["subjects_test"]
    random.seed(args.seed)
    chosen = sorted(random.sample(train_pool, args.n)) if args.n < len(train_pool) else train_pool

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    tr = DataLoader(ThighSliceDataset(subject_files(chosen, cache5), augment=True), args.bs, True,
                    num_workers=args.workers, pin_memory=True, drop_last=len(chosen) > 1)
    va = DataLoader(ThighSliceDataset(subject_files(val_subj, cache5)), args.bs, False, num_workers=args.workers, pin_memory=True)
    te = DataLoader(ThighSliceDataset(subject_files(test_subj, cache5)), args.bs, False, num_workers=args.workers, pin_memory=True)

    model = DixonFuseUNet(args.variant, nc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = DiceCELoss(nc).to(dev)
    scaler = torch.cuda.amp.GradScaler()

    rp = "" if args.region == "thigh" else f"{args.region}_"
    tag = f"method_{rp}{args.variant}_n{args.n}_s{args.seed}"
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
    # capture average learned contrast gate on test set (interpretability)
    gate = None
    if args.variant == "dixonfuse":
        gs = []
        model.eval()
        with torch.no_grad():
            for img, _ in te:
                model(img.to(dev)); gs.append(model.attn.last_gate.mean(0).cpu().numpy())
        gate = np.mean(gs, 0).round(4).tolist()
    res = {"variant": args.variant, "n_subjects": args.n, "seed": args.seed,
           "classes": meta["classes"], "channel_names": meta["channel_names"],
           "test_dice_per_class": [None if np.isnan(x) else round(float(x), 4) for x in td],
           "test_dice_fg_mean": round(float(np.nanmean(td[1:])), 4),
           "test_iou_fg_mean": round(float(np.nanmean(ti[1:])), 4),
           "learned_contrast_gate": gate,
           "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(os.path.join(C.RESULTS, tag + ".json"), "w"), indent=2)
    print(f"[{tag}] DONE dice_fg={res['test_dice_fg_mean']} gate={gate} ({res['minutes']}min)", flush=True)


if __name__ == "__main__":
    main()
