"""Generic region prep for lower-limb Dixon MRI (Thigh or Calf).
Builds BOTH the 4-channel and 5-channel (+fat-fraction) 2D slice caches and the
subject split/meta, mirroring prep_thigh*.py but parameterised by region.

Usage:  python prep_region.py --region Calf
Outputs (region lower-cased as tag, e.g. 'calf'):
  data_cache/<tag>_slices/   , data_cache/<tag>_slices5/
  data_cache/<tag>_meta.json , <tag>_split.json  (4ch)
  data_cache/<tag>5_meta.json, <tag>5_split.json (5ch)
"""
import os, glob, json, argparse, random
import numpy as np
import nibabel as nib
from skimage.transform import resize as _rs
import config as C

CONTRASTS = ["Water", "Fat", "In_phase", "Opp_phase"]
H = W = 256


def load(subj, region):
    d = os.path.join(C.THIGH_ROOT, subj, region)
    vols = {c: nib.load(os.path.join(d, c + ".nii.gz")).get_fdata() for c in CONTRASTS}
    m = nib.load(os.path.join(d, "mask_muscles.nii.gz")).get_fdata().astype(np.int64)
    return vols, m


def z(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=["Thigh", "Calf"])
    args = ap.parse_args()
    region = args.region; tag = region.lower()
    label_json = json.load(open(os.path.join(C.DATA_ROOT, "Thigh",
                            f"{tag}_muscle_segmentation_labels.json")))
    subs = sorted(os.listdir(C.THIGH_ROOT))

    # contiguous relabel from union
    all_labels = set()
    for s in subs:
        _, m = load(s, region); all_labels |= set(np.unique(m).tolist())
    labels = sorted(all_labels); remap = {int(v): i for i, v in enumerate(labels)}
    nc = len(labels)
    names = [label_json["labels"].get(str(o), {}).get("name", f"label_{o}") for o in labels]

    dir4 = os.path.join(C.CACHE, f"{tag}_slices"); os.makedirs(dir4, exist_ok=True)
    dir5 = os.path.join(C.CACHE, f"{tag}_slices5"); os.makedirs(dir5, exist_ok=True)
    per_subj = {}
    for s in subs:
        vols, m = load(s, region)
        m = np.vectorize(remap.get)(m).astype(np.int64)
        zc = {c: z(vols[c]) for c in CONTRASTS}
        rawW, rawF = vols["Water"], vols["Fat"]
        ff = np.clip(rawF / (rawF + rawW + 1e-6), 0, 1)
        files = []
        for zz in range(m.shape[2]):
            msk = m[:, :, zz]
            if msk.max() == 0:
                continue
            msk_r = _rs(msk, (H, W), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
            ch4 = np.stack([zc[c][:, :, zz] for c in CONTRASTS], 0)
            ch4r = np.stack([_rs(ch4[k], (H, W), order=1, preserve_range=True) for k in range(4)], 0).astype(np.float32)
            np.savez_compressed(os.path.join(dir4, f"{s}_{zz:03d}.npz"), img=ch4r, mask=msk_r)
            ch5 = np.concatenate([ch4, ff[None, :, :, zz]], 0)
            ch5r = np.stack([_rs(ch5[k], (H, W), order=1, preserve_range=True) for k in range(5)], 0).astype(np.float32)
            np.savez_compressed(os.path.join(dir5, f"{s}_{zz:03d}.npz"), img=ch5r, mask=msk_r)
            files.append(f"{s}_{zz:03d}.npz")
        per_subj[s] = files
        print(f"{s}: {len(files)} slices", flush=True)

    random.seed(C.SEED); order = subs[:]; random.shuffle(order)
    n = len(order); n_test = max(1, round(0.2 * n)); n_val = max(1, round(0.1 * n))
    test, val, train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    meta_base = {"num_classes": nc, "classes": names, "orig_labels": labels,
                 "subjects_train": train, "subjects_val": val, "subjects_test": test,
                 "H": H, "W": W, "region": region}
    for suff, d, inch, chn in [("", dir4, 4, None),
                               ("5", dir5, 5, ["Water", "Fat", "In_phase", "Opp_phase", "FatFraction"])]:
        split = {k: [os.path.join(d, f) for s in meta_base[f"subjects_{k}"] for f in per_subj[s]]
                 for k in ["train", "val", "test"]}
        json.dump(split, open(os.path.join(C.CACHE, f"{tag}{suff}_split.json"), "w"))
        meta = dict(meta_base); meta["in_channels"] = inch
        if chn: meta["channel_names"] = chn
        json.dump(meta, open(os.path.join(C.CACHE, f"{tag}{suff}_meta.json"), "w"), indent=2)
    print(f"[{tag}] classes={nc} train/val/test={len(train)}/{len(val)}/{len(test)} "
          f"slices={sum(len(v) for v in per_subj.values())}")


if __name__ == "__main__":
    main()
