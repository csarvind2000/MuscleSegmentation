"""Cache the external (HuashanMyo) cohort in the PRIMARY model's 4-channel input space
and label space, for source-model fine-tuning (R1.1, option 2).

Reuses zeroshot_external.subject_entries (single-thigh FOV crop, 4ch [W,F,In,Opp],
256x256). Saves per-slice npz with:
  img       : 4xHxW float32  (matches the primary thigh model)
  mask_prim : HxW int64      (external GT remapped to primary muscle indices, for loss)
  mask_ext  : HxW int64      (original HuashanMyo GT, for the 10-muscle eval metric)
Writes data_cache/ext4prim/<sid>_<i>.npz
"""
import os, json
import numpy as np
import config as C
from zeroshot_external import subject_entries

# HuashanMyo label -> primary model class (inverse of MUSCLE_MAP; BL & BB both -> biceps 7)
EXT2PRIM = {1: 5, 2: 1, 3: 2, 4: 3, 5: 4, 6: 12, 7: 6, 8: 7, 9: 8, 10: 9, 11: 7}
OUT = os.path.join(C.CACHE, "ext4prim")


def remap(mask_ext):
    m = np.zeros_like(mask_ext, dtype=np.int64)
    for e, p in EXT2PRIM.items():
        m[mask_ext == e] = p
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(C.CACHE, "ext_meta.json")))
    sids = meta["subjects_train"] + meta["subjects_val"] + meta["subjects_test"]
    total = 0
    for sid in sids:
        r = subject_entries(sid)
        if not r or not r["entries"]:
            print(f"{sid}: no entries"); continue
        for i, e in enumerate(r["entries"]):
            np.savez_compressed(os.path.join(OUT, f"{sid}_{i:03d}.npz"),
                                img=e["img"].astype(np.float32),
                                mask_prim=remap(e["mask"]),
                                mask_ext=e["mask"].astype(np.int64))
        total += len(r["entries"])
        print(f"{sid}: {len(r['entries'])} slices", flush=True)
    print(f"cached {total} slices for {len(sids)} subjects -> {OUT}")


if __name__ == "__main__":
    main()
