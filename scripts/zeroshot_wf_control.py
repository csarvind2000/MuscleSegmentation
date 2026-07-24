"""R1 control: isolate cross-cohort shift from channel-reconstruction.

Trains a primary-cohort thigh model on ONLY water+fat (2-channel), identically to the
4-channel benchmark otherwise, then applies it zero-shot to HuashanMyo using its NATIVE
water+fat (NO in/opposed-phase reconstruction). Comparing this to the 4-channel zero-shot
(0.09-0.17 Dice, which fed reconstructed In=W+F, Opp=|W-F|) separates the two effects.

Writes results/zeroshot_wf_control.json
"""
import os, glob, json, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config as C
from models import build_model
from datasets import _aug2d_multi
import zeroshot_external as Z          # reuse subject_entries/predict_subject/subject_dice/MUSCLE_MAP/z/EXT

WF = [0, 1]                            # water, fat (order matches prep_thigh CONTRASTS)


class WFThigh(Dataset):
    def __init__(self, files, augment=False):
        self.files, self.augment = files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = d["img"].astype(np.float32)[WF]     # keep only water+fat
        mask = d["mask"].astype(np.int64)
        if self.augment:
            img, mask = _aug2d_multi(img, mask)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


def dicece(logits, target, nc):
    ce = torch.nn.functional.cross_entropy(logits, target)
    probs = torch.softmax(logits, 1)
    t1h = torch.nn.functional.one_hot(target, nc).permute(0, 3, 1, 2).float()
    inter = (probs * t1h).sum((0, 2, 3)); denom = probs.sum((0, 2, 3)) + t1h.sum((0, 2, 3))
    return ce + (1 - ((2 * inter + 1) / (denom + 1)).mean())


def train_wf(epochs=80):
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    split = json.load(open(os.path.join(C.CACHE, "thigh_split.json")))
    nc = meta["num_classes"]; dev = "cuda"
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    tr = DataLoader(WFThigh(split["train"], True), batch_size=8, shuffle=True,
                    num_workers=6, pin_memory=True, drop_last=True)
    va = DataLoader(WFThigh(split["val"]), batch_size=8, num_workers=4)
    net = build_model("unet_r34", 2, nc).to(dev)            # 2-ch input, ImageNet init (as benchmark)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler()
    ckpt = os.path.join(C.CKPT, "thigh_unet_r34_wf.pt"); best = -1; t0 = time.time()
    for ep in range(epochs):
        net.train()
        for img, mask in tr:
            img, mask = img.to(dev), mask.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = dicece(net(img), mask, nc)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        net.eval(); ds = np.zeros(nc); dc = np.zeros(nc)
        with torch.no_grad():
            for img, mask in va:
                pr = net(img.to(dev)).argmax(1).cpu().numpy(); m = mask.numpy()
                for c in range(1, nc):
                    p = (pr == c); t = (m == c); dn = p.sum() + t.sum()
                    if dn > 0: ds[c] += 2 * (p & t).sum() / dn; dc[c] += 1
        vfg = float(np.nanmean(np.where(dc > 0, ds / np.maximum(dc, 1), np.nan)[1:]))
        if vfg > best: best = vfg; torch.save(net.state_dict(), ckpt)
        print(f"[wf-train] ep{ep+1}/{epochs} val_fg={vfg:.4f}", flush=True)
    print(f"[wf-train] done best_val_fg={best:.4f} in {(time.time()-t0)/60:.1f} min", flush=True)
    return ckpt, round(best, 4)


def zeroshot_wf(ckpt):
    dev = "cuda"
    net = build_model("unet_r34", 2, 14, encoder_weights=None)
    net.load_state_dict(torch.load(ckpt, map_location="cpu")); net = net.to(dev).eval()
    wf_chan = lambda Wv, Fv: [Z.z(Wv), Z.z(Fv)]            # native 2-channel, NO reconstruction
    sids = sorted({os.path.basename(f).replace(".nii.gz", "")
                   for f in glob.glob(os.path.join(Z.EXT, "Label", "*.nii.gz"))})
    per_subj = []
    for sid in sids:
        s = Z.subject_entries(sid, chan_fn=wf_chan)
        if not s or not s["entries"]:
            continue
        slices = [(e["img"], e["mask"]) for e in s["entries"]]
        preds = Z.predict_subject(net, slices, dev)
        d = Z.subject_dice(preds, [g for _, g in slices])
        if d:
            per_subj.append(float(np.mean(list(d.values()))))
    a = np.array(per_subj)
    print(f"[wf-zeroshot] n={len(a)} mean={a.mean():.4f} sd={a.std(ddof=1):.4f} "
          f"range={a.min():.3f}-{a.max():.3f}", flush=True)
    return a


if __name__ == "__main__":
    ck, bestval = train_wf(80)
    a = zeroshot_wf(ck)
    json.dump({"protocol": "primary trained on water+fat (2ch); zero-shot to HuashanMyo native "
                           "water+fat, NO in/opp reconstruction",
               "primary_val_fg_dice": bestval,
               "n": len(a), "mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4),
               "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
               "per_subject": [round(float(x), 4) for x in a]},
              open(os.path.join(C.RESULTS, "zeroshot_wf_control.json"), "w"), indent=2)
    print("wrote results/zeroshot_wf_control.json", flush=True)
