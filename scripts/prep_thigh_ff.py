"""Re-cache thigh MRI slices with a 5th physics channel: fat-fraction (FF).

FF = rawFat / (rawFat + rawWater + eps), computed from RAW intensities (before
z-scoring), clipped to [0,1]. Channels stored: [Water_z, Fat_z, In_z, Opp_z, FF].
Writes to data_cache/thigh_slices5/ and thigh5_{split,meta}.json.
Subject splits are identical to the 4-channel cache (reuse thigh_meta.json).
"""
import os, glob, json
import numpy as np
import nibabel as nib
from skimage.transform import resize as _rs
import config as C

REGION = "Thigh"
CONTRASTS = ["Water", "Fat", "In_phase", "Opp_phase"]
MASK = "mask_muscles.nii.gz"
OUT = os.path.join(C.CACHE, "thigh_slices5")
H = W = 256


def load(subj):
    d = os.path.join(C.THIGH_ROOT, subj, REGION)
    vols = {c: nib.load(os.path.join(d, c + ".nii.gz")).get_fdata() for c in CONTRASTS}
    m = nib.load(os.path.join(d, MASK)).get_fdata().astype(np.int64)
    return vols, m


def z(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def main():
    os.makedirs(OUT, exist_ok=True)
    base = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    labels = base["orig_labels"]
    remap = {int(v): i for i, v in enumerate(labels)}
    subs = sorted(os.listdir(C.THIGH_ROOT))
    per_subj = {}
    for s in subs:
        vols, m = load(s)
        m = np.vectorize(remap.get)(m).astype(np.int64)
        raw_w, raw_f = vols["Water"], vols["Fat"]
        ff = raw_f / (raw_f + raw_w + 1e-6)
        ff = np.clip(ff, 0, 1)
        ch = [z(vols["Water"]), z(vols["Fat"]), z(vols["In_phase"]), z(vols["Opp_phase"]), ff]
        files = []
        for zz in range(m.shape[2]):
            msk = m[:, :, zz]
            if msk.max() == 0:
                continue
            img = np.stack([c[:, :, zz] for c in ch], 0)
            img_r = np.stack([_rs(img[k], (H, W), order=1, preserve_range=True) for k in range(img.shape[0])], 0).astype(np.float32)
            msk_r = _rs(msk, (H, W), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
            fp = os.path.join(OUT, f"{s}_{zz:03d}.npz")
            np.savez_compressed(fp, img=img_r, mask=msk_r)
            files.append(fp)
        per_subj[s] = files
        print(f"{s}: {len(files)} slices", flush=True)

    split = {k: [f for s in base[f"subjects_{k}"] for f in per_subj[s]]
             for k in ["train", "val", "test"]}
    json.dump(split, open(os.path.join(C.CACHE, "thigh5_split.json"), "w"))
    meta = dict(base); meta["in_channels"] = 5
    meta["channel_names"] = ["Water", "Fat", "In_phase", "Opp_phase", "FatFraction"]
    json.dump(meta, open(os.path.join(C.CACHE, "thigh5_meta.json"), "w"), indent=2)
    print("slices:", {k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()
