"""Isolate the effect of through-plane slice thickness on a thin-slice-trained model.

The primary cohort has 5 mm slices; the external cohort has 26 mm slices. We take the
trained primary model and apply it to primary TEST data in which every input slice is the
average of k adjacent 5 mm slices (k=1 -> 5 mm native; k=5 -> ~25 mm, close to the external
26 mm), keeping the CENTRE-slice label as ground truth. Only the input's through-plane
averaging changes, so any Dice drop is attributable to slice thickness alone.

Writes results/slice_thickness_ablation.json
"""
import os, json, glob
import numpy as np
import torch
import config as C
from models import build_model

VOL = os.path.join(C.CACHE, "thigh_vol")


def thick(vol, z, k):
    D = vol.shape[-1]; h = k // 2
    zs = list(range(max(0, z - h), min(D, z + h + 1)))
    return vol[:, :, :, zs].mean(-1)          # (4,H,W)


def main():
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    nc = meta["num_classes"]
    net = build_model("unet_r34", 4, nc, encoder_weights=None)
    net.load_state_dict(torch.load(os.path.join(C.CKPT, "thigh_unet_r34.pt"), map_location="cpu"))
    net = net.cuda().eval()

    subj = {s: np.load(os.path.join(VOL, f"{s}.npz")) for s in meta["subjects_test"]}
    out = {}
    for k in [1, 3, 5, 7]:
        means = []
        for s, d in subj.items():
            vol = d["vol"].astype(np.float32); mask = d["mask"]; zlab = d["zlab"].tolist()
            inter = np.zeros(nc); den = np.zeros(nc)
            for z in zlab:
                img = thick(vol, z, k)
                with torch.no_grad():
                    pr = net(torch.from_numpy(img[None]).cuda()).argmax(1)[0].cpu().numpy()
                gt = mask[:, :, z]
                for c in range(nc):
                    p = (pr == c); t = (gt == c); inter[c] += (p & t).sum(); den[c] += p.sum() + t.sum()
            dd = np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan)
            means.append(float(np.nanmean(dd[1:])))
        out[f"k{k}_{k*5}mm"] = round(float(np.mean(means)), 4)
        print(f"k={k}  ~{k*5:2d} mm slices: per-subject Dice = {out[f'k{k}_{k*5}mm']}", flush=True)
    json.dump(out, open(os.path.join(C.RESULTS, "slice_thickness_ablation.json"), "w"), indent=2)
    print("drop 5mm->25mm:", round(out["k1_5mm"] - out["k5_25mm"], 3))


if __name__ == "__main__":
    main()
