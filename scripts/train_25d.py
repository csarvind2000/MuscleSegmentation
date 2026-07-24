"""2.5D thigh muscle segmentation baseline (reviewer R1.2/R1.4).

Provides inter-slice context by stacking +/-k adjacent slices (each 4-channel Dixon) as
input to the SAME U-Net/ResNet-34 recipe used for the 2D baseline; the network predicts
the centre slice. k=0 reproduces the 2D model. Uses the identical subject split as the 2D
baseline (thigh_meta.json) for a fair head-to-head, and the same DiceCE loss and metrics.

Usage: python train_25d.py --model unet_r34 --k 1 --epochs 80
Writes results/thigh_25d_k<k>_<model>.json
"""
import os, json, glob, argparse, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config as C
from models import build_model
from train import DiceCELoss, dice_score, iou_score
from datasets import _aug2d_multi

VOL = os.path.join(C.CACHE, "thigh_vol")


class Vol25(Dataset):
    """Stacks +/-k neighbour slices (clamped at volume ends) -> (4*(2k+1), H, W)."""
    def __init__(self, subjects, k, augment=False):
        self.k = k; self.augment = augment
        self.vols = {}; self.index = []
        for s in subjects:
            d = np.load(os.path.join(VOL, f"{s}.npz"))
            self.vols[s] = (d["vol"], d["mask"])
            for z in d["zlab"].tolist():
                self.index.append((s, int(z)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, z = self.index[i]
        vol, mask = self.vols[s]                       # (4,H,W,D),(H,W,D)
        D = vol.shape[-1]
        zs = [min(max(z + off, 0), D - 1) for off in range(-self.k, self.k + 1)]
        img = np.concatenate([vol[:, :, :, zz] for zz in zs], 0).astype(np.float32)  # 4*(2k+1),H,W
        m = mask[:, :, z].astype(np.int64)
        if self.augment:
            img, m = _aug2d_multi(img, m)
        return torch.from_numpy(img).float(), torch.from_numpy(m).long()


@torch.no_grad()
def evaluate(model, loader, nc, dev):
    model.eval(); dsum = np.zeros(nc); dcnt = np.zeros(nc); isum = np.zeros(nc); icnt = np.zeros(nc)
    for img, mask in loader:
        pred = model(img.to(dev)).argmax(1).cpu()
        for b in range(pred.shape[0]):
            for c, v in enumerate(dice_score(pred[b:b+1], mask[b:b+1], nc)):
                if not np.isnan(v): dsum[c] += v; dcnt[c] += 1
            for c, v in enumerate(iou_score(pred[b:b+1], mask[b:b+1], nc)):
                if not np.isnan(v): isum[c] += v; icnt[c] += 1
    dice = np.where(dcnt > 0, dsum / np.maximum(dcnt, 1), np.nan)
    iou = np.where(icnt > 0, isum / np.maximum(icnt, 1), np.nan)
    return dice, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_r34")
    ap.add_argument("--k", type=int, default=1, help="neighbours each side; 0 = 2D")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    nc = meta["num_classes"]; classes = meta["classes"]; in_ch = 4 * (2 * args.k + 1)

    tr = DataLoader(Vol25(meta["subjects_train"], args.k, augment=True), args.bs, True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    va = DataLoader(Vol25(meta["subjects_val"], args.k), args.bs, False, num_workers=args.workers, pin_memory=True)
    te = DataLoader(Vol25(meta["subjects_test"], args.k), args.bs, False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, in_ch, nc, encoder_weights="imagenet").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = DiceCELoss(nc).to(dev); scaler = torch.cuda.amp.GradScaler()

    best = -1; tmp = os.path.join(C.CKPT, f"_25d_tmp_k{args.k}_{args.model}.pt"); t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for img, mask in tr:
            img, mask = img.to(dev), mask.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        vd, _ = evaluate(model, va, nc, dev); vfg = float(np.nanmean(vd[1:]))
        if vfg > best:
            best = vfg; torch.save(model.state_dict(), tmp)
    model.load_state_dict(torch.load(tmp)); os.remove(tmp)
    dice, iou = evaluate(model, te, nc, dev)
    # per-SUBJECT volume Dice on the test subjects (comparable to the subject-level Table 3)
    model.eval(); subj_means = []
    with torch.no_grad():
        for s in meta["subjects_test"]:
            ds = Vol25([s], args.k)
            if len(ds) == 0:
                continue
            inter = np.zeros(nc); den = np.zeros(nc)
            for i in range(len(ds)):
                img, m = ds[i]
                pr = model(img[None].to(dev)).argmax(1)[0].cpu().numpy(); gt = m.numpy()
                for c in range(nc):
                    p = (pr == c); t = (gt == c); inter[c] += (p & t).sum(); den[c] += p.sum() + t.sum()
            d = np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan)
            subj_means.append(float(np.nanmean(d[1:])))
    persubj = round(float(np.mean(subj_means)), 4)
    res = {"task": "thigh", "model": args.model, "k": args.k, "mode": f"2.5D(±{args.k})" if args.k else "2D",
           "in_channels": in_ch, "num_classes": nc, "classes": classes,
           "test_dice_per_class": [round(float(x), 4) for x in dice],
           "test_dice_fg_mean": round(float(np.nanmean(dice[1:])), 4),
           "test_dice_persubject": persubj,
           "test_iou_fg_mean": round(float(np.nanmean(iou[1:])), 4),
           "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "minutes": round((time.time() - t0) / 60, 1)}
    out = os.path.join(C.RESULTS, f"thigh_25d_k{args.k}_{args.model}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"[thigh 2.5D k={args.k} {args.model}] per-slice={res['test_dice_fg_mean']} "
          f"per-subject={res['test_dice_persubject']} ({res['minutes']}min)", flush=True)


if __name__ == "__main__":
    main()
