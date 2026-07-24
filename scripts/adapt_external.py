"""Source-model fine-tuning on the external cohort (R1.1, option 2).

Fine-tune the PRIMARY-trained thigh model (init='primary') on N target subjects and
evaluate on the fixed 10-subject test set with the SAME 10-muscle metric as the zero-shot
test, so the result anchors at N=0 = the zero-shot number (~0.11). An ImageNet-from-scratch
run on the identical 4-channel/primary-label pipeline (init='imagenet') is the matched
control, isolating the benefit of source knowledge over ImageNet.

Usage: python adapt_external.py --init primary --n 4 --seed 0
Writes results/ext_adapt_<init>_n<N>_s<seed>.json
"""
import os, json, glob, argparse, time, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config as C
from models import build_model
from train import DiceCELoss
from datasets import _aug2d_multi
from zeroshot_external import subject_dice, MUSCLE_MAP

CACHE = os.path.join(C.CACHE, "ext4prim")


class Ext4(Dataset):
    def __init__(self, files, augment=False):
        self.files = files; self.augment = augment
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = d["img"].astype(np.float32); mask = d["mask_prim"].astype(np.int64)
        if self.augment:
            img, mask = _aug2d_multi(img, mask)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


def sfiles(subjects):
    return [f for s in subjects for f in sorted(glob.glob(os.path.join(CACHE, f"{s}_*.npz")))]


@torch.no_grad()
def eval_subjects(model, subjects, dev):
    model.eval(); means = []
    for s in subjects:
        files = sorted(glob.glob(os.path.join(CACHE, f"{s}_*.npz")))
        if not files:
            continue
        preds, gts = [], []
        for f in files:
            d = np.load(f)
            x = torch.from_numpy(d["img"][None]).float().to(dev)
            preds.append(model(x).argmax(1).cpu().numpy()[0])
            gts.append(d["mask_ext"])
        dd = subject_dice(np.stack(preds), gts)   # 10-muscle metric (primary pred vs ext GT)
        if dd:
            means.append(float(np.mean(list(dd.values()))))
    return float(np.mean(means)) if means else float("nan")


def build(init, dev):
    ew = "imagenet" if init == "imagenet" else None
    net = build_model("unet_r34", 4, 14, encoder_weights=ew)
    if init == "primary":
        net.load_state_dict(torch.load(os.path.join(C.CKPT, "thigh_unet_r34.pt"), map_location="cpu"))
    return net.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, choices=["primary", "imagenet"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(C.CACHE, "ext_meta.json")))
    pool, val, test = meta["subjects_train"], meta["subjects_val"], meta["subjects_test"]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    chosen = sorted(random.sample(pool, args.n)) if args.n < len(pool) else pool
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = DataLoader(Ext4(sfiles(chosen), augment=True), args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=len(chosen) > 1)
    model = build(args.init, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = DiceCELoss(14).to(dev); scaler = torch.cuda.amp.GradScaler()

    best = -1; tmp = os.path.join(C.CKPT, f"_adapt_tmp_{args.init}_{args.seed}.pt"); t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for img, mask in tr:
            img, mask = img.to(dev), mask.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        vfg = eval_subjects(model, val, dev)
        if vfg > best:
            best = vfg; torch.save(model.state_dict(), tmp)
    model.load_state_dict(torch.load(tmp)); os.remove(tmp)
    per = {}
    for s in test:
        files = sorted(glob.glob(os.path.join(CACHE, f"{s}_*.npz")))
        preds = []; gts = []
        for f in files:
            d = np.load(f)
            with torch.no_grad():
                preds.append(model(torch.from_numpy(d["img"][None]).float().to(dev)).argmax(1).cpu().numpy()[0])
            gts.append(d["mask_ext"])
        dd = subject_dice(np.stack(preds), gts)
        per[s] = round(float(np.mean(list(dd.values()))), 4) if dd else None
    overall = round(float(np.mean([v for v in per.values() if v is not None])), 4)
    tag = f"ext_adapt_{args.init}_n{args.n}_s{args.seed}"
    json.dump({"init": args.init, "n": args.n, "seed": args.seed, "metric": "10-muscle zeroshot",
               "test_dice_mean": overall, "per_subject": per,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(C.RESULTS, tag + ".json"), "w"), indent=2)
    print(f"[{tag}] dice={overall} ({round((time.time()-t0)/60,1)}min)", flush=True)


if __name__ == "__main__":
    main()
