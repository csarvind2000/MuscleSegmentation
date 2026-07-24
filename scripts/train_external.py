"""External-cohort validation trainer (multi-ethnic HuashanMyo, 66 subjects).

Tests whether the two core findings replicate on an independent cohort:
  - pretraining source:  --init random|imagenet|ct   (Water,Fat,FF = 3ch)
  - fat-fraction effect:  --channels wf (Water,Fat, 2ch)  vs  wff (Water,Fat,FF, 3ch)
Trains on N subjects (subject-level), evaluates per-subject Dice on the fixed test set.

Usage: python train_external.py --init imagenet --channels wff --n 8 --seed 0
Writes results/ext_<channels>_<init>_n<N>_s<seed>.json
"""
import os, json, glob, argparse, time, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import config as C
from datasets import ThighSliceDataset
from train import DiceCELoss, dice_score


def sfiles(subjects, cache):
    return [f for s in subjects for f in sorted(glob.glob(os.path.join(cache, f"{s}_*.npz")))]


def adapt_ct(net, in_ch):
    sd = torch.load(os.path.join(C.CKPT, "aattct_unet_r34.pt"), map_location="cpu")
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    tgt = net.encoder.state_dict()
    for k, v in enc.items():
        if k not in tgt:
            continue
        if v.shape == tgt[k].shape:
            tgt[k] = v
        elif v.dim() == 4 and v.shape[1] == 1 and tgt[k].shape[1] == in_ch:
            tgt[k] = v.repeat(1, in_ch, 1, 1) / in_ch
    net.encoder.load_state_dict(tgt); return net


class Sel(nn.Module):
    def __init__(s, net, ch): super().__init__(); s.net = net; s.ch = ch
    def forward(s, x): return s.net(x[:, :s.ch])


def build(init, in_ch, nc):
    ew = "imagenet" if init in ("imagenet", "ct") else None
    net = smp.Unet("resnet34", encoder_weights=ew, in_channels=in_ch, classes=nc)
    if init == "ct":
        adapt_ct(net, in_ch)
    elif init == "ssl":
        # SSL encoder self-supervised on the target cohort's own unlabelled images.
        # Encoder was pretrained with 3 input channels (wff); matches in_ch there.
        sd = torch.load(os.path.join(C.CKPT, "ssl_encoder_ext_wff.pt"), map_location="cpu")
        tgt = net.encoder.state_dict()
        for k, v in sd.items():
            if k in tgt and v.shape == tgt[k].shape:
                tgt[k] = v
            elif k in tgt and v.dim() == 4 and v.shape[1] == 3 and tgt[k].shape[1] == in_ch:
                tgt[k] = v[:, :in_ch]  # allow wf (2ch) reuse of the 3ch SSL stem
        net.encoder.load_state_dict(tgt)
    return net


@torch.no_grad()
def per_subject(model, subjects, cache, nc, dev):
    out = {}
    for s in subjects:
        dl = DataLoader(ThighSliceDataset(sorted(glob.glob(os.path.join(cache, f"{s}_*.npz")))), 8, False, num_workers=4)
        inter = np.zeros(nc); den = np.zeros(nc)
        for img, mask in dl:
            pred = model(img.to(dev)).argmax(1).cpu()
            for c in range(nc):
                p = (pred == c); t = (mask == c); inter[c] += (p & t).sum().item(); den[c] += p.sum().item() + t.sum().item()
        d = np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan)
        out[s] = round(float(np.nanmean(d[1:])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, choices=["random", "imagenet", "ct", "ssl"])
    ap.add_argument("--channels", required=True, choices=["wf", "wff"])
    ap.add_argument("--n", type=int, default=0)  # 0 = all training subjects
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cache = os.path.join(C.CACHE, "ext_slices3")
    meta = json.load(open(os.path.join(C.CACHE, "ext_meta.json")))
    nc = meta["num_classes"]; in_ch = 2 if args.channels == "wf" else 3
    pool = meta["subjects_train"]; val = meta["subjects_val"]; test = meta["subjects_test"]
    random.seed(args.seed)
    chosen = sorted(random.sample(pool, args.n)) if args.n and args.n < len(pool) else pool
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = "cuda"

    tr = DataLoader(SelDS := ThighSliceDataset(sfiles(chosen, cache), augment=True), args.bs, True,
                    num_workers=args.workers, pin_memory=True, drop_last=len(chosen) > 1)
    va = DataLoader(ThighSliceDataset(sfiles(val, cache)), args.bs, False, num_workers=args.workers, pin_memory=True)
    model = Sel(build(args.init, in_ch, nc), in_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = DiceCELoss(nc).to(dev); scaler = torch.cuda.amp.GradScaler()
    best = -1; t0 = time.time()
    from train import evaluate
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
            best = vfg; torch.save(model.state_dict(), os.path.join(C.CKPT, "_ext_tmp.pt"))
    model.load_state_dict(torch.load(os.path.join(C.CKPT, "_ext_tmp.pt")))
    model.eval()
    ps = per_subject(model, test, cache, nc, dev)
    nn_ = args.n if args.n else len(pool)
    tag = f"ext_{args.channels}_{args.init}_n{nn_}_s{args.seed}"
    json.dump({"init": args.init, "channels": args.channels, "n": nn_, "seed": args.seed,
               "test_dice_fg_mean": round(float(np.mean(list(ps.values()))), 4),
               "per_subject_dice": ps, "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(C.RESULTS, tag + ".json"), "w"), indent=2)
    print(f"[{tag}] dice={round(float(np.mean(list(ps.values()))),4)} ({round((time.time()-t0)/60,1)}min)", flush=True)


if __name__ == "__main__":
    main()
