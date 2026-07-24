"""Option B: fat-fraction-anchored, label-free cross-cohort domain adaptation.

The zero-shot external gap is largely an intensity/contrast difference between scanners.
The Dixon fat-fraction FF = F/(F+W) is a physical, scanner-invariant quantity, so we use
it to normalise water and fat into a common, scanner-independent space: within each volume
we locate near-pure fat (FF>0.85) and near-pure muscle (FF<0.15) using FF alone (no
labels), and linearly rescale each channel so muscle and fat map to fixed values. The
in/opposed-phase channels are rebuilt from the normalised water and fat. Both cohorts are
normalised identically, so a model trained on the primary cohort should transfer better
zero-shot, with no target labels used at any point.

Trains a U-Net on the primary cohort under FF-anchored normalisation and evaluates it
zero-shot on all 66 HuashanMyo subjects (10 shared muscles), for comparison with the
standard zero-shot baseline (~0.11).

Usage: python option_b_ffa.py --epochs 80
Writes results/ext_ffanchor_zeroshot.json
"""
import os, glob, json, argparse, time
import numpy as np
import nibabel as nib
from skimage.transform import resize as _rs
import torch
from torch.utils.data import Dataset, DataLoader
import config as C
from models import build_model
from train import DiceCELoss
from datasets import _aug2d_multi
import zeroshot_external as Z

REGION = "Thigh"
H = W = 256


def ff_anchor(Wv, Fv):
    """FF-landmark normalisation. Returns 4 channel volumes [Wn, Fn, In, Opp]."""
    Wv = Wv.astype(np.float32); Fv = Fv.astype(np.float32)
    denom = Wv + Fv
    FF = np.clip(Fv / (denom + 1e-6), 0, 1)
    pos = denom[denom > 0]
    thr = 0.10 * np.percentile(pos, 99) if pos.size else 0.0
    fg = denom > thr
    fat_m = fg & (FF > 0.85)      # near-pure fat
    mus_m = fg & (FF < 0.15)      # near-pure muscle
    def norm(X, bright, dark):
        hi = np.median(X[bright]) if bright.sum() > 100 else np.percentile(X[fg], 95)
        lo = np.median(X[dark]) if dark.sum() > 100 else np.percentile(X[fg], 5)
        return ((X - lo) / (hi - lo + 1e-6)).astype(np.float32)
    Wn = norm(Wv, mus_m, fat_m)   # water: bright in muscle, dark in fat
    Fn = norm(Fv, fat_m, mus_m)   # fat:   bright in fat, dark in muscle
    return [Wn, Fn, (Wn + Fn).astype(np.float32), np.abs(Wn - Fn).astype(np.float32)]


class PrimarySlices(Dataset):
    """Primary thigh labelled slices under FF-anchored normalisation, held in memory."""
    def __init__(self, subjects, remap, augment=False):
        self.items = []; self.augment = augment
        for s in subjects:
            d = os.path.join(C.THIGH_ROOT, s, REGION)
            Wv = nib.load(os.path.join(d, "Water.nii.gz")).get_fdata()
            Fv = nib.load(os.path.join(d, "Fat.nii.gz")).get_fdata()
            m = nib.load(os.path.join(d, "mask_muscles.nii.gz")).get_fdata().astype(np.int64)
            m = np.vectorize(remap.get)(m).astype(np.int64)
            ch = ff_anchor(Wv, Fv)
            for k in range(m.shape[2]):
                if m[:, :, k].max() == 0:
                    continue
                img = np.stack([_rs(c[:, :, k], (H, W), order=1, preserve_range=True) for c in ch], 0).astype(np.float32)
                msk = _rs(m[:, :, k], (H, W), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
                self.items.append((img, msk))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img, m = self.items[i]
        if self.augment:
            img, m = _aug2d_multi(img.copy(), m.copy())
        return torch.from_numpy(img).float(), torch.from_numpy(m).long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--model", default="unet_r34")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    nc = meta["num_classes"]; labels = meta["orig_labels"]; remap = {int(v): i for i, v in enumerate(labels)}
    train_subj = meta["subjects_train"] + meta["subjects_val"]   # train on all non-test primary subjects

    print("building FF-anchored primary training set...", flush=True)
    ds = PrimarySlices(train_subj, remap, augment=True)
    dl = DataLoader(ds, args.bs, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    print(f"  {len(ds)} slices from {len(train_subj)} subjects", flush=True)

    model = build_model(args.model, 4, nc, encoder_weights="imagenet").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = DiceCELoss(nc).to(dev); scaler = torch.cuda.amp.GradScaler()
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for img, mask in dl:
            img, mask = img.to(dev), mask.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    torch.save(model.state_dict(), os.path.join(C.CKPT, "thigh_ffanchor_unet.pt"))
    model.eval()

    # zero-shot on all 66 external subjects under the SAME FF-anchored normalisation
    print("zero-shot on HuashanMyo (FF-anchored)...", flush=True)
    sids = sorted({os.path.basename(f).replace(".nii.gz", "") for f in glob.glob(os.path.join(Z.EXT, "Label", "*.nii.gz"))})
    per = {}
    for sid in sids:
        subj = Z.subject_entries(sid, chan_fn=ff_anchor)
        if not subj or not subj["entries"]:
            continue
        slices = [(e["img"], e["mask"]) for e in subj["entries"]]
        pred = Z.predict_subject(model, slices, dev)
        dd = Z.subject_dice(pred, [e["mask"] for e in subj["entries"]])
        if dd:
            per[sid] = float(np.mean(list(dd.values())))
    overall = float(np.mean(list(per.values())))
    out = {"method": "ff_anchored_zeroshot", "model": args.model,
           "overall_mean_dice": round(overall, 4),
           "overall_std_dice": round(float(np.std(list(per.values()))), 4),
           "baseline_zeroshot": 0.111, "per_subject": {k: round(v, 4) for k, v in per.items()},
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(os.path.join(C.RESULTS, "ext_ffanchor_zeroshot.json"), "w"), indent=2)
    print(f"[FF-anchored zero-shot] mean Dice = {overall:.4f}  (baseline {out['baseline_zeroshot']}) "
          f"({out['minutes']}min)", flush=True)


if __name__ == "__main__":
    main()
