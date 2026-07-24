"""Generic 2D segmentation trainer + evaluator (Dice/IoU per class, HD95 optional).

Usage:
  python train.py --task aattct  --model unet_r34 --epochs 40
  python train.py --task thigh   --model unet_r34 --epochs 60
Outputs a json of test metrics under results/ and best checkpoint under checkpoints/.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from models import build_model, BASELINES
from datasets import AATTCTDataset, ThighSliceDataset


def dice_score(pred, target, num_classes, eps=1e-6):
    """pred,target: (N,H,W) long. Returns per-class dice list (ignores class absent in both)."""
    dices = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        if denom == 0:
            dices.append(float("nan"))
        else:
            dices.append((2 * inter + eps) / (denom + eps))
    return dices


def iou_score(pred, target, num_classes, eps=1e-6):
    ious = []
    for c in range(num_classes):
        p = (pred == c); t = (target == c)
        inter = (p & t).sum().item(); union = (p | t).sum().item()
        ious.append(float("nan") if union == 0 else (inter + eps) / (union + eps))
    return ious


class DiceCELoss(nn.Module):
    def __init__(self, num_classes, ce_weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.nc = num_classes

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, 1)
        t1h = torch.nn.functional.one_hot(target, self.nc).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * t1h).sum(dims)
        denom = probs.sum(dims) + t1h.sum(dims)
        dice = (2 * inter + 1) / (denom + 1)
        return ce + (1 - dice.mean())


def make_loaders(task, bs, workers):
    if task == "aattct":
        idx = os.path.join(C.CACHE, "aattct_index.csv")
        tr = AATTCTDataset(idx, "train", augment=True)
        va = AATTCTDataset(idx, "val")
        te = AATTCTDataset(idx, "test")
        in_ch, nc = 1, C.AATTCT_NUM_CLASSES
        classes = C.AATTCT_CLASSES
    elif task in ("thigh", "calf"):
        meta = json.load(open(os.path.join(C.CACHE, f"{task}_meta.json")))
        split = json.load(open(os.path.join(C.CACHE, f"{task}_split.json")))
        tr = ThighSliceDataset(split["train"], augment=True)
        va = ThighSliceDataset(split["val"])
        te = ThighSliceDataset(split["test"])
        in_ch, nc = meta["in_channels"], meta["num_classes"]
        classes = meta["classes"]
    else:
        raise ValueError(task)
    dl = lambda ds, sh: DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=workers,
                                   pin_memory=True, drop_last=sh)
    return dl(tr, True), dl(va, False), dl(te, False), in_ch, nc, classes


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    dsum = np.zeros(num_classes); dcnt = np.zeros(num_classes)
    isum = np.zeros(num_classes); icnt = np.zeros(num_classes)
    for img, mask in loader:
        img = img.to(device); mask = mask.to(device)
        logits = model(img)
        pred = logits.argmax(1)
        for d in dice_score(pred, mask, num_classes):
            pass
        # accumulate per-image to handle nan (absent classes)
        for b in range(pred.shape[0]):
            for c, v in enumerate(dice_score(pred[b:b+1], mask[b:b+1], num_classes)):
                if not np.isnan(v): dsum[c] += v; dcnt[c] += 1
            for c, v in enumerate(iou_score(pred[b:b+1], mask[b:b+1], num_classes)):
                if not np.isnan(v): isum[c] += v; icnt[c] += 1
    dice = np.where(dcnt > 0, dsum / np.maximum(dcnt, 1), np.nan)
    iou = np.where(icnt > 0, isum / np.maximum(icnt, 1), np.nan)
    return dice, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["aattct", "thigh", "calf"])
    ap.add_argument("--model", required=True, choices=list(BASELINES))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te, in_ch, nc, classes = make_loaders(args.task, args.bs, args.workers)
    # imagenet weights need 3ch; smp adapts in_channels automatically
    model = build_model(args.model, in_ch, nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = DiceCELoss(nc).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    tag = f"{args.task}_{args.model}"
    best_val = -1; best_path = os.path.join(C.CKPT, f"{tag}.pt")
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); running = 0.0
        for img, mask in tr:
            img = img.to(device); mask = mask.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running += loss.item()
        sched.step()
        vdice, _ = evaluate(model, va, nc, device)
        vfg = float(np.nanmean(vdice[1:]))  # mean foreground dice
        hist.append({"epoch": ep, "train_loss": running / len(tr), "val_dice_fg": vfg})
        print(f"[{tag}] ep{ep+1}/{args.epochs} loss={running/len(tr):.4f} val_dice_fg={vfg:.4f}", flush=True)
        if vfg > best_val:
            best_val = vfg
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path))
    tdice, tiou = evaluate(model, te, nc, device)
    res = {
        "task": args.task, "model": args.model, "epochs": args.epochs,
        "num_classes": nc, "classes": classes,
        "test_dice_per_class": [None if np.isnan(x) else round(float(x), 4) for x in tdice],
        "test_iou_per_class": [None if np.isnan(x) else round(float(x), 4) for x in tiou],
        "test_dice_fg_mean": round(float(np.nanmean(tdice[1:])), 4),
        "test_iou_fg_mean": round(float(np.nanmean(tiou[1:])), 4),
        "best_val_dice_fg": round(float(best_val), 4),
        "minutes": round((time.time() - t0) / 60, 1),
        "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "history": hist,
    }
    out = os.path.join(C.RESULTS, f"{tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"[{tag}] DONE test_dice_fg={res['test_dice_fg_mean']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
