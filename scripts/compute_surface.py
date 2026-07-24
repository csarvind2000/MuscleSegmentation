"""Thigh boundary metrics (HD95, ASSD, surface Dice @2px) for a trained model, matching
Table 4 / Supplementary S4: per slice over foreground muscles on the 256 grid, averaged
over held-out subjects. Validate on unet_r34 (expect ~4.77 / 1.26 / 0.903).

Usage: python compute_surface.py --model unet_hrnet18
"""
import os, glob, json, argparse
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
import config as C
from models import build_model


def surf(pr, gt, tol=2.0):
    if gt.sum() == 0 or pr.sum() == 0:
        return None
    gb = gt ^ binary_erosion(gt)
    pb = pr ^ binary_erosion(pr)
    if gb.sum() == 0 or pb.sum() == 0:
        return None
    dtg = distance_transform_edt(~gb); dtp = distance_transform_edt(~pb)
    d_pg = dtg[pb]; d_gp = dtp[gb]
    alld = np.concatenate([d_pg, d_gp])
    hd95 = np.percentile(alld, 95); assd = alld.mean()
    sdice = (np.sum(d_pg <= tol) + np.sum(d_gp <= tol)) / (len(d_pg) + len(d_gp))
    return hd95, assd, sdice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    nc = meta["num_classes"]; test = meta["subjects_test"]
    cache = os.path.join(C.CACHE, "thigh_slices")
    net = build_model(args.model, meta["in_channels"], nc, encoder_weights=None)
    net.load_state_dict(torch.load(os.path.join(C.CKPT, f"thigh_{args.model}.pt"), map_location="cpu"))
    net = net.to(dev).eval()

    subj_hd, subj_as, subj_sd = [], [], []
    with torch.no_grad():
        for s in test:
            hd, asd, sd = [], [], []
            for f in sorted(glob.glob(os.path.join(cache, f"{s}_*.npz"))):
                d = np.load(f)
                pred = net(torch.from_numpy(d["img"][None]).float().to(dev)).argmax(1).cpu().numpy()[0]
                gt = d["mask"]
                for c in range(1, nc):
                    r = surf(pred == c, gt == c)
                    if r:
                        hd.append(r[0]); asd.append(r[1]); sd.append(r[2])
            if hd:
                subj_hd.append(np.mean(hd)); subj_as.append(np.mean(asd)); subj_sd.append(np.mean(sd))
    res = {"model": args.model, "hd95_px": round(float(np.mean(subj_hd)), 2),
           "assd_px": round(float(np.mean(subj_as)), 2), "sdice2px": round(float(np.mean(subj_sd)), 3)}
    json.dump(res, open(os.path.join(C.RESULTS, f"surf_thigh_{args.model}.json"), "w"), indent=2)
    print(f"[{args.model}] HD95={res['hd95_px']}  ASSD={res['assd_px']}  sDice@2px={res['sdice2px']}")


if __name__ == "__main__":
    main()
