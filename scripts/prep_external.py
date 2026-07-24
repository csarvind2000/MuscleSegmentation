"""Preprocess the external multi-ethnic thigh-muscle cohort (HuashanMyo, 66 subjects)
for external validation. Uses Water and Fat (IDEAL/Dixon) plus a fat-fraction proxy.

Caches 2D slices as 3-channel [Water_z, Fat_z, FF] with the 11-muscle label
(12 classes incl. background), resized to 256x256. Writes data_cache/ext_slices3/
and ext_meta.json / ext_split.json (subject-level 70/15/15).
"""
import os, glob, json, random
import numpy as np
import nibabel as nib
from skimage.transform import resize as _rs
import config as C

EXT = "/media/ranjhaa-local/volume23/sarcopenia/datasets/multiethnic_thigh/HuashanMyo/HuashanMyo"
OUT = os.path.join(C.CACHE, "ext_slices3")
H = W = 256


def z(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def main():
    os.makedirs(OUT, exist_ok=True)
    labs = sorted(glob.glob(os.path.join(EXT, "Label", "*.nii.gz")))
    # union of labels
    allu = set()
    for f in labs:
        allu |= set(np.unique(nib.load(f).get_fdata().astype(int)).tolist())
    labels = sorted(allu); remap = {int(v): i for i, v in enumerate(labels)}
    nc = len(labels)

    per_subj = {}
    for lf in labs:
        sid = os.path.basename(lf).replace(".nii.gz", "")
        wf = os.path.join(EXT, "Water", sid + "_0001.nii.gz")
        ff_ = os.path.join(EXT, "Fat", sid + "_0000.nii.gz")
        if not (os.path.exists(wf) and os.path.exists(ff_)):
            continue
        W_ = nib.load(wf).get_fdata(); F_ = nib.load(ff_).get_fdata()
        m = np.vectorize(remap.get)(nib.load(lf).get_fdata().astype(int)).astype(np.int64)
        ff = np.clip(F_ / (F_ + W_ + 1e-6), 0, 1)
        wz, fz = z(W_), z(F_)
        files = []
        for k in range(m.shape[2]):
            msk = m[:, :, k]
            if msk.max() == 0:
                continue
            img = np.stack([_rs(wz[:, :, k], (H, W), order=1, preserve_range=True),
                            _rs(fz[:, :, k], (H, W), order=1, preserve_range=True),
                            _rs(ff[:, :, k], (H, W), order=1, preserve_range=True)], 0).astype(np.float32)
            msk_r = _rs(msk, (H, W), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
            fp = os.path.join(OUT, f"{sid}_{k:03d}.npz")
            np.savez_compressed(fp, img=img, mask=msk_r); files.append(fp)
        per_subj[sid] = files
        print(f"{sid}: {len(files)} slices", flush=True)

    subs = sorted(per_subj); random.seed(C.SEED); random.shuffle(subs)
    n = len(subs); nte = round(0.15 * n); nva = round(0.15 * n)
    test, val, train = subs[:nte], subs[nte:nte + nva], subs[nte + nva:]
    split = {k: [f for s in v for f in per_subj[s]] for k, v in
             {"train": train, "val": val, "test": test}.items()}
    json.dump(split, open(os.path.join(C.CACHE, "ext_split.json"), "w"))
    json.dump({"num_classes": nc, "in_channels": 3, "orig_labels": labels,
               "channel_names": ["Water", "Fat", "FatFraction"],
               "subjects_train": train, "subjects_val": val, "subjects_test": test,
               "n_subjects": n}, open(os.path.join(C.CACHE, "ext_meta.json"), "w"), indent=2)
    print(f"[ext] subjects={n} classes={nc} train/val/test={len(train)}/{len(val)}/{len(test)} "
          f"slices={sum(len(v) for v in per_subj.values())}")


if __name__ == "__main__":
    main()
