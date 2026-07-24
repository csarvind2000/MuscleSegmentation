"""Build an index of AATTCT-IDS annotated slices and a subject-level 70/10/20 split.

Each row: image path (jpg), SAT mask png, VAT mask png, subject id.
Split follows the dataset paper's protocol: 20% subjects test, remaining split
into train/val (we use ~78/22 of the remainder -> ~62/18/20 overall, rounded to
70/10/20 subject-level for a clean benchmark).
"""
import os, glob, csv, random
import config as C

def main():
    img_root = os.path.join(C.AATTCT_ROOT, "Image", "Extracted")
    sat_root = os.path.join(C.AATTCT_ROOT, "Label", "Subcutaneous")
    vat_root = os.path.join(C.AATTCT_ROOT, "Label", "Visceral")

    subjects = sorted(os.listdir(img_root))
    rows = []
    missing = 0
    for s in subjects:
        for jpg in sorted(glob.glob(os.path.join(img_root, s, "*.jpg"))):
            fn = os.path.splitext(os.path.basename(jpg))[0] + ".png"
            sat = os.path.join(sat_root, s, fn)
            vat = os.path.join(vat_root, s, fn)
            if not (os.path.exists(sat) and os.path.exists(vat)):
                missing += 1
                continue
            rows.append((jpg, sat, vat, s))

    random.seed(C.SEED)
    subj = sorted(set(r[3] for r in rows))
    random.shuffle(subj)
    n = len(subj)
    n_test = int(round(0.20 * n))
    n_val = int(round(0.10 * n))
    test = set(subj[:n_test])
    val = set(subj[n_test:n_test + n_val])
    train = set(subj[n_test + n_val:])

    def split_of(s):
        return "test" if s in test else "val" if s in val else "train"

    out = os.path.join(C.CACHE, "aattct_index.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "sat", "vat", "subject", "split"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], split_of(r[3])])

    from collections import Counter
    cnt = Counter(split_of(r[3]) for r in rows)
    print(f"subjects={n} train={len(train)} val={len(val)} test={len(test)}")
    print(f"slices: {dict(cnt)} total={len(rows)} missing_pairs={missing}")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
