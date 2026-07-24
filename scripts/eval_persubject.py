"""Per-subject per-muscle Dice (volume-level) for a trained full-split model, matching the
protocol of Supplementary Table S4 (tab:s_muscle_dice). Used to add HRNet to the
architecture comparison on the SAME footing as the existing 2D models.

Evaluates a checkpoint on the held-out test subjects: for each subject, pool a muscle's
voxels across all its (labelled) slices -> one Dice per muscle per subject; average over
subjects. Reports the per-muscle vector and the foreground mean.

Usage: python eval_persubject.py --region thigh --model unet_hrnet18
       python eval_persubject.py --region thigh --model unet_r34   # validation
"""
import os, glob, json, argparse
import numpy as np
import torch
import config as C
from models import build_model

CACHE = {"thigh": "thigh_slices", "calf": "calf_slices"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="thigh", choices=["thigh", "calf"])
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(C.CACHE, f"{args.region}_meta.json")))
    nc = meta["num_classes"]; classes = meta["classes"]; in_ch = meta["in_channels"]
    test = meta["subjects_test"]
    cache = os.path.join(C.CACHE, CACHE[args.region])

    net = build_model(args.model, in_ch, nc, encoder_weights=None)
    net.load_state_dict(torch.load(os.path.join(C.CKPT, f"{args.region}_{args.model}.pt"), map_location="cpu"))
    net = net.to(dev).eval()

    per_subj = {c: [] for c in range(1, nc)}
    with torch.no_grad():
        for s in test:
            files = sorted(glob.glob(os.path.join(cache, f"{s}_*.npz")))
            if not files:
                continue
            inter = np.zeros(nc); den = np.zeros(nc)
            for f in files:
                d = np.load(f)
                x = torch.from_numpy(d["img"][None]).float().to(dev)
                pred = net(x).argmax(1).cpu().numpy()[0]
                gt = d["mask"]
                for c in range(nc):
                    p = (pred == c); t = (gt == c)
                    inter[c] += (p & t).sum(); den[c] += p.sum() + t.sum()
            for c in range(1, nc):
                if den[c] > 0:
                    per_subj[c].append(2 * inter[c] / den[c])
    permuscle = {classes[c]: round(float(np.mean(per_subj[c])), 4) for c in range(1, nc) if per_subj[c]}
    fg = round(float(np.mean(list(permuscle.values()))), 4)
    out = {"region": args.region, "model": args.model, "protocol": "per-subject volume Dice, held-out",
           "per_muscle": permuscle, "fg_mean": fg}
    json.dump(out, open(os.path.join(C.RESULTS, f"persubj_{args.region}_{args.model}.json"), "w"), indent=2)
    print(f"[{args.region} {args.model}] per-subject fg mean Dice = {fg}")
    for k, v in permuscle.items():
        print(f"    {k:20s} {v:.4f}")


if __name__ == "__main__":
    main()
