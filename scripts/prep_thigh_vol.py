"""Cache per-subject thigh volumes for the 2.5D baseline (reviewer R1.2/R1.4).

The 2D cache (prep_thigh.py) keeps only muscle-containing slices, so neighbouring slices
needed for 2.5D context may be missing. Here we cache the full z-scored 4-channel volume
and the remapped label volume per subject, plus the list of labelled slice indices. The
2.5D dataset then stacks +/-k neighbours around each labelled centre slice.

Writes data_cache/thigh_vol/<subject>.npz  {vol(4,H,W,D) f16, mask(H,W,D) i8, zlab list}
Reuses the subject split / class mapping from thigh_meta.json (same as the 2D baseline).
"""
import os, json
import numpy as np
import nibabel as nib
from skimage.transform import resize as _rs
import config as C

REGION = "Thigh"
CONTRASTS = ["Water", "Fat", "In_phase", "Opp_phase"]
MASK = "mask_muscles.nii.gz"
OUT = os.path.join(C.CACHE, "thigh_vol")
H = W = 256


def z(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(C.CACHE, "thigh_meta.json")))
    labels = meta["orig_labels"]; remap = {int(v): i for i, v in enumerate(labels)}
    subs = meta["subjects_train"] + meta["subjects_val"] + meta["subjects_test"]
    for s in subs:
        d = os.path.join(C.THIGH_ROOT, s, REGION)
        vols = [z(nib.load(os.path.join(d, c + ".nii.gz")).get_fdata().astype(np.float32)) for c in CONTRASTS]
        m = nib.load(os.path.join(d, MASK)).get_fdata().astype(np.int64)
        m = np.vectorize(remap.get)(m).astype(np.int8)
        D = m.shape[2]
        vol = np.stack([np.stack([_rs(v[:, :, k], (H, W), order=1, preserve_range=True) for k in range(D)], -1)
                        for v in vols], 0).astype(np.float16)                 # 4,H,W,D
        mask = np.stack([_rs(m[:, :, k], (H, W), order=0, preserve_range=True, anti_aliasing=False)
                         for k in range(D)], -1).astype(np.int8)              # H,W,D
        zlab = [k for k in range(D) if mask[:, :, k].max() > 0]
        np.savez_compressed(os.path.join(OUT, f"{s}.npz"), vol=vol, mask=mask, zlab=np.array(zlab))
        print(f"{s}: D={D} labelled={len(zlab)}", flush=True)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
