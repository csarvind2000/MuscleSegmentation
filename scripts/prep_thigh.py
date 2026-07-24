"""Audit + cache the Thigh MRI muscle-segmentation task as 2D slices.

Modes:
  python prep_thigh.py --audit     # print shapes, contrasts, label values
  python prep_thigh.py             # build 2D slice cache + subject split + meta

Input per subject: MRI_data/<id>/Thigh/{Water,Fat,In_phase,Opp_phase}.nii.gz
Label: mask_muscles.nii.gz  (multi-class individual thigh muscles)
We cache only slices that contain >=1 labeled muscle voxel.
Channels used as input: Water, Fat, In_phase, Opp_phase (4ch), z-normalized per volume.
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
import config as C

REGION = "Thigh"
CONTRASTS = ["Water", "Fat", "In_phase", "Opp_phase"]
MASK = "mask_muscles.nii.gz"
CACHE_DIR = os.path.join(C.CACHE, "thigh_slices")


def load(subj):
    d = os.path.join(C.THIGH_ROOT, subj, REGION)
    vols = {c: nib.load(os.path.join(d, c + ".nii.gz")).get_fdata() for c in CONTRASTS}
    m = nib.load(os.path.join(d, MASK)).get_fdata().astype(np.int64)
    return vols, m


def audit():
    subs = sorted(os.listdir(C.THIGH_ROOT))
    all_labels = set()
    for s in subs:
        try:
            vols, m = load(s)
        except Exception as e:
            print(f"{s}: ERR {e}"); continue
        shp = vols["Water"].shape
        u = np.unique(m)
        all_labels |= set(u.tolist())
        if s in subs[:3] or s == subs[-1]:
            print(f"{s}: shape={shp} mask_uniq={u.tolist()}")
    print("\nUNION of mask labels across subjects:", sorted(all_labels))
    print("num subjects:", len(subs))


def zscore(v):
    mu, sd = v.mean(), v.std()
    return (v - mu) / (sd + 1e-6)


def build():
    os.makedirs(CACHE_DIR, exist_ok=True)
    subs = sorted(os.listdir(C.THIGH_ROOT))
    # relabel to contiguous ids
    all_labels = set()
    for s in subs:
        _, m = load(s)
        all_labels |= set(np.unique(m).tolist())
    labels = sorted(all_labels)
    remap = {int(v): i for i, v in enumerate(labels)}  # 0 stays background
    num_classes = len(labels)
    print("labels:", labels, "-> num_classes", num_classes)

    per_subj_files = {}
    H = W = 256  # resize target for consistent batching
    from skimage.transform import resize as _rs
    for s in subs:
        vols, m = load(s)
        m = np.vectorize(remap.get)(m).astype(np.int64)
        vch = [zscore(vols[c]) for c in CONTRASTS]  # each HxWxD
        D = m.shape[2]
        files = []
        for z in range(D):
            msk = m[:, :, z]
            if msk.max() == 0:
                continue
            img = np.stack([v[:, :, z] for v in vch], 0)  # 4xHxW
            # resize
            img_r = np.stack([_rs(img[c], (H, W), order=1, preserve_range=True) for c in range(img.shape[0])], 0).astype(np.float32)
            msk_r = _rs(msk, (H, W), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
            fp = os.path.join(CACHE_DIR, f"{s}_{z:03d}.npz")
            np.savez_compressed(fp, img=img_r, mask=msk_r)
            files.append(fp)
        per_subj_files[s] = files
        print(f"{s}: cached {len(files)} slices")

    # subject-level split 70/10/20
    import random
    random.seed(C.SEED)
    order = subs[:]; random.shuffle(order)
    n = len(order); n_test = max(1, round(0.2 * n)); n_val = max(1, round(0.1 * n))
    test = order[:n_test]; val = order[n_test:n_test + n_val]; train = order[n_test + n_val:]
    split = {"train": [f for s in train for f in per_subj_files[s]],
             "val":   [f for s in val for f in per_subj_files[s]],
             "test":  [f for s in test for f in per_subj_files[s]]}
    json.dump(split, open(os.path.join(C.CACHE, "thigh_split.json"), "w"))
    # class names
    label_json = json.load(open(os.path.join(C.DATA_ROOT, "Thigh", "thigh_muscle_segmentation_labels.json")))
    names = []
    for orig in labels:
        info = label_json["labels"].get(str(orig), {})
        names.append(info.get("name", f"label_{orig}"))
    meta = {"in_channels": len(CONTRASTS), "num_classes": num_classes,
            "classes": names, "orig_labels": labels,
            "subjects_train": train, "subjects_val": val, "subjects_test": test,
            "H": H, "W": W}
    json.dump(meta, open(os.path.join(C.CACHE, "thigh_meta.json"), "w"), indent=2)
    print("split subjects:", {"train": len(train), "val": len(val), "test": len(test)})
    print("slices:", {k: len(v) for k, v in split.items()})
    print("classes:", names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    audit() if args.audit else build()
