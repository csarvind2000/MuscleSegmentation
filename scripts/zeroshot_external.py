"""TRUE zero-shot external test (reviewer R1.1).

Train on the primary 25-subject Wohlfarth thigh dataset, then apply the resulting
model *directly* to the independent HuashanMyo cohort (66 subjects) with NO retraining
and NO fine-tuning. This replaces the previous "external" experiment, which re-split
HuashanMyo and trained within it (a second-dataset experiment, not external validation).

Channel matching:  the primary model expects 4 channels [Water, Fat, In-phase, Opp-phase].
HuashanMyo provides Water and Fat (2-point Dixon magnitude images); the in/opposed-phase
channels are reconstructed as In = Water + Fat and Opp = |Water - Fat|. Every channel is
z-scored per volume and each slice resized to 256x256, matching prep_thigh.py exactly.

Label matching:  the primary model outputs 14 classes (bg + 13 muscles); HuashanMyo labels
11 muscles. We score the 10 anatomically-corresponding muscles. HuashanMyo splits biceps
femoris into long (8) + short (11) heads, which are merged to match the primary single
"biceps_femoris" class. Primary-only muscles (adductor brevis/longus, gluteus maximus)
have no external ground truth and are not scored.

Usage:
  python zeroshot_external.py --models unet_r34 unetpp_r34 deeplabv3p_r34 segformer_b1
  python zeroshot_external.py --models unet_r34 --limit 2 --qc   # quick dry run + QC PNG
"""
import os, glob, json, argparse, time
import numpy as np
import nibabel as nib
from skimage.transform import resize as _rs
import torch
import config as C
from models import build_model

EXT = ("/media/ranjhaa-local/volume23/sarcopenia/datasets/opensource_dataset/"
       "multiethnic_thigh/HuashanMyo/HuashanMyo")
H = W = 256

# primary_model_class_index -> (muscle name, {set of HuashanMyo label indices})
MUSCLE_MAP = {
    1:  ("rectus_femoris",    {2}),
    2:  ("vastus_lateralis",  {3}),
    3:  ("vastus_intermedius",{4}),
    4:  ("vastus_medialis",   {5}),
    5:  ("sartorius",         {1}),
    6:  ("gracilis",          {7}),
    7:  ("biceps_femoris",    {8, 11}),   # long + short head merged
    8:  ("semitendinosus",    {9}),
    9:  ("semimembranosus",   {10}),
    12: ("adductor_magnus",   {6}),
}


def z(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def _fg_bbox(fg2d, margin=0.08):
    """Bounding box of a 2D foreground mask, expanded by `margin`. None if empty."""
    ys, xs = np.where(fg2d)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my = int(round((y1 - y0) * margin)); mx = int(round((x1 - x0) * margin))
    return (max(0, y0 - my), min(fg2d.shape[0], y1 + my + 1),
            max(0, x0 - mx), min(fg2d.shape[1], x1 + mx + 1))


# primary model class index -> HuashanMyo label index, so saved predictions share the
# GT's integer scheme. Primary-only muscles (adductor brevis/longus, gluteus) have no
# external counterpart and are dropped to background in the saved masks.
PRIM2EXT = {1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 7, 7: 8, 8: 9, 9: 10, 12: 6}


def subject_entries(sid, split=True, chan_fn=None):
    """Preprocess one HuashanMyo subject to match the primary single-thigh FOV.

    Bilateral volumes are split into the two thighs (along the axis with the clearest
    foreground gap); each thigh is cropped to its intensity foreground bbox (labels are
    NOT used for cropping) and resized to 256x256, matching prep_thigh.py. Returns a dict
    with per-thigh-slice entries plus the geometry needed to paste predictions back into
    the original volume, and the source affine for saving NIfTI.
    """
    wf = os.path.join(EXT, "Water", f"{sid}_0001.nii.gz")
    ff = os.path.join(EXT, "Fat",   f"{sid}_0000.nii.gz")
    lf = os.path.join(EXT, "Label", f"{sid}.nii.gz")
    if not (os.path.exists(wf) and os.path.exists(ff) and os.path.exists(lf)):
        return None
    wimg = nib.load(wf)
    Wv = wimg.get_fdata().astype(np.float32)
    Fv = nib.load(ff).get_fdata().astype(np.float32)
    Lv = nib.load(lf).get_fdata().astype(np.int64)
    if chan_fn is not None:                       # Option B: FF-anchored normalization
        chans = chan_fn(Wv, Fv)
    else:
        In = Wv + Fv
        Op = np.abs(Wv - Fv)
        chans = [z(Wv), z(Fv), z(In), z(Op)]      # order matches prep_thigh CONTRASTS

    fg_vol = Wv > (0.10 * Wv.max())               # intensity foreground (no labels used)
    H0, W0, D = Wv.shape

    if split:
        rowprof = fg_vol.sum(axis=(1, 2)); colprof = fg_vol.sum(axis=(0, 2))
        def gap(prof):
            n = len(prof); b0, b1 = int(n * 0.35), int(n * 0.65)
            return (prof.max() - prof[b0:b1].min()) / (prof.max() + 1e-6), b0 + int(np.argmin(prof[b0:b1]))
        gr, sr = gap(rowprof); gc, sc = gap(colprof)
        regions = [("rows", 0, sr), ("rows", sr, H0)] if gr >= gc else [("cols", 0, sc), ("cols", sc, W0)]
    else:
        regions = [("rows", 0, H0)]

    entries = []
    for axis, a0, a1 in regions:
        for k in range(D):
            if axis == "rows":
                sub_fg = fg_vol[a0:a1, :, k]; sub_lab = Lv[a0:a1, :, k]
                sub_ch = [c[a0:a1, :, k] for c in chans]
            else:
                sub_fg = fg_vol[:, a0:a1, k]; sub_lab = Lv[:, a0:a1, k]
                sub_ch = [c[:, a0:a1, k] for c in chans]
            if sub_lab.max() == 0:
                continue
            bb = _fg_bbox(sub_fg)
            if bb is None:
                continue
            y0, y1, x0, x1 = bb
            img = np.stack([_rs(c[y0:y1, x0:x1], (H, W), order=1, preserve_range=True) for c in sub_ch], 0).astype(np.float32)
            msk_r = _rs(sub_lab[y0:y1, x0:x1], (H, W), order=0, preserve_range=True, anti_aliasing=False).astype(np.int64)
            entries.append({"img": img, "mask": msk_r,
                            "place": (axis, a0, a1, y0, y1, x0, x1), "k": k})
    return {"entries": entries, "shape": (H0, W0, D),
            "affine": wimg.affine, "header": wimg.header}


def load_subject(sid, split=True):
    """Back-compat wrapper: list of (img 4xHxW, extmask HxW) entries."""
    r = subject_entries(sid, split)
    return None if r is None else [(e["img"], e["mask"]) for e in r["entries"]]


def reconstruct_volume(subj, preds):
    """Paste per-thigh 256x256 predictions (primary label space) back into the original
    volume geometry, remapped to the HuashanMyo label scheme. Returns (H0,W0,D) int array."""
    H0, W0, D = subj["shape"]
    vol = np.zeros((H0, W0, D), dtype=np.int16)
    for e, pred in zip(subj["entries"], preds):
        axis, a0, a1, y0, y1, x0, x1 = e["place"]
        ext = np.zeros_like(pred, dtype=np.int16)
        for pi, ei in PRIM2EXT.items():
            ext[pred == pi] = ei
        h, w = (y1 - y0), (x1 - x0)
        back = _rs(ext, (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(np.int16)
        if axis == "rows":
            vol[a0:a1, :, e["k"]][y0:y1, x0:x1] = back
        else:
            vol[:, a0:a1, e["k"]][y0:y1, x0:x1] = back
    return vol


@torch.no_grad()
def predict_subject(model, slices, dev, bs=8):
    preds = []
    for i in range(0, len(slices), bs):
        batch = np.stack([s[0] for s in slices[i:i + bs]], 0)
        x = torch.from_numpy(batch).float().to(dev)
        preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds, 0)               # (Nslices, H, W) in PRIMARY label space


def subject_dice(pred, gts):
    """Volume-level Dice per common muscle for one subject. Returns {name: dice}."""
    out = {}
    gt = np.stack(gts, 0)                          # (Nslices, H, W) external labels
    for prim_idx, (name, ext_set) in MUSCLE_MAP.items():
        p = (pred == prim_idx)
        t = np.isin(gt, list(ext_set))
        denom = p.sum() + t.sum()
        if denom == 0:
            continue
        out[name] = float(2 * (p & t).sum() / denom)
    return out


def load_model(name, dev):
    net = build_model(name, 4, 14, encoder_weights=None)
    sd = torch.load(os.path.join(C.CKPT, f"thigh_{name}.pt"), map_location="cpu")
    net.load_state_dict(sd)
    return net.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["unet_r34"])
    ap.add_argument("--limit", type=int, default=0, help="only first N subjects (dry run)")
    ap.add_argument("--qc", action="store_true", help="save a QC overlay PNG for the first subject")
    ap.add_argument("--save_nifti", action="store_true",
                    help="save predicted mask volumes (HuashanMyo label scheme) as NIfTI per subject/model")
    ap.add_argument("--tag", default="zeroshot")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sids = sorted({os.path.basename(f).replace(".nii.gz", "")
                   for f in glob.glob(os.path.join(EXT, "Label", "*.nii.gz"))})
    if args.limit:
        sids = sids[:args.limit]
    print(f"subjects: {len(sids)}  device: {dev}")

    # cache preprocessed entries once (shared across models); keeps geometry for NIfTI export
    print("preprocessing external volumes...", flush=True)
    cache = {}
    t0 = time.time()
    for sid in sids:
        s = subject_entries(sid)
        if s and s["entries"]:
            cache[sid] = s
    print(f"  preprocessed {len(cache)} subjects in {time.time()-t0:.1f}s", flush=True)

    results = {}
    for mname in args.models:
        model = load_model(mname, dev)
        nifti_dir = os.path.join(C.RESULTS, "zeroshot_masks", mname)
        if args.save_nifti:
            os.makedirs(nifti_dir, exist_ok=True)
        per_subj = {}
        for sid, subj in cache.items():
            slices = [(e["img"], e["mask"]) for e in subj["entries"]]
            pred = predict_subject(model, slices, dev)
            gts = [e["mask"] for e in subj["entries"]]
            per_subj[sid] = subject_dice(pred, gts)
            if args.save_nifti:
                vol = reconstruct_volume(subj, list(pred))
                out = nib.Nifti1Image(vol, subj["affine"], subj["header"])
                out.set_data_dtype(np.int16)
                nib.save(out, os.path.join(nifti_dir, f"{sid}_pred.nii.gz"))
        # aggregate: per-muscle mean across subjects, and per-subject mean-over-muscles
        muscles = [n for _, (n, _) in MUSCLE_MAP.items()]
        permuscle = {n: float(np.mean([per_subj[s][n] for s in per_subj if n in per_subj[s]]))
                     for n in muscles}
        subj_means = [float(np.mean(list(per_subj[s].values()))) for s in per_subj if per_subj[s]]
        overall = float(np.mean(subj_means))
        results[mname] = {"overall_mean_dice": round(overall, 4),
                          "overall_std_dice": round(float(np.std(subj_means)), 4),
                          "per_muscle": {k: round(v, 4) for k, v in permuscle.items()},
                          "per_subject": {s: {k: round(v, 4) for k, v in d.items()}
                                          for s, d in per_subj.items()}}
        print(f"\n=== {mname} :: zero-shot HuashanMyo ===")
        print(f"  overall mean Dice (subject-level): {overall:.4f} +/- {results[mname]['overall_std_dice']:.4f}")
        for n in muscles:
            print(f"    {n:20s} {permuscle[n]:.4f}")

        if args.qc:
            _save_qc(model, cache, dev, mname, args.tag)

    outp = os.path.join(C.RESULTS, f"ext_{args.tag}.json")
    json.dump(results, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}")

    if args.save_nifti:
        legend = {1: "Sartorius", 2: "Rectus femoris", 3: "Vastus lateralis",
                  4: "Vastus intermedius", 5: "Vastus medialis", 6: "Adductor magnus",
                  7: "Gracilis", 8: "Biceps femoris (long+short, merged)",
                  9: "Semitendinosus", 10: "Semimembranosus"}
        mdir = os.path.join(C.RESULTS, "zeroshot_masks")
        with open(os.path.join(mdir, "LABELS_legend.txt"), "w") as f:
            f.write("Predicted zero-shot masks — HuashanMyo label scheme.\n")
            f.write("Overlay <sid>_pred.nii.gz on the original Water/Fat/Label volumes.\n")
            f.write("Predictions are in original volume geometry; only the 10 muscles the\n")
            f.write("primary model shares with HuashanMyo are emitted (others -> background).\n")
            f.write("Note HuashanMyo GT uses 8=BL,11=BB for the two biceps heads; the model\n")
            f.write("has a single biceps class, saved here as 8.\n\n")
            for k, v in legend.items():
                f.write(f"{k}\t{v}\n")
        print(f"NIfTI masks under {mdir}/<model>/<sid>_pred.nii.gz  (+ LABELS_legend.txt)")


def _save_qc(model, cache, dev, mname, tag):
    """Overlay GT vs prediction on the middle muscle slice of the first subject (orientation sanity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sid = next(iter(cache))
    slices = cache[sid]
    k = len(slices) // 2
    img, gt = slices[k]
    with torch.no_grad():
        pred = model(torch.from_numpy(img[None]).float().to(dev)).argmax(1).cpu().numpy()[0]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img[0], cmap="gray"); ax[0].set_title(f"{sid} Water")
    ax[1].imshow(gt, cmap="nipy_spectral"); ax[1].set_title("GT (HuashanMyo labels)")
    ax[2].imshow(pred, cmap="nipy_spectral"); ax[2].set_title(f"pred ({mname}, primary labels)")
    for a in ax: a.axis("off")
    p = os.path.join(C.FIG, f"qc_zeroshot_{mname}_{tag}.png")
    plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
    print(f"  QC saved: {p}")


if __name__ == "__main__":
    main()
