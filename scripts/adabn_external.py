"""AdaBN source-free domain adaptation for the external zero-shot test.

Takes the primary-trained 4-channel thigh model and recomputes its BatchNorm running
statistics on the external HuashanMyo IMAGES ONLY (no labels, no gradient updates), then
re-evaluates. This is unsupervised / source-free domain adaptation, NOT zero-shot: it uses
the target images (but never the target labels). Reports baseline zero-shot Dice and the
AdaBN-adapted Dice on the same 66 subjects.

Writes results/ext_adabn.json
"""
import os, glob, json, time
import numpy as np
import torch
import torch.nn as nn
import config as C
import zeroshot_external as Z


def build_ext_cache():
    sids = sorted({os.path.basename(f).replace(".nii.gz", "")
                   for f in glob.glob(os.path.join(Z.EXT, "Label", "*.nii.gz"))})
    cache = {}
    for sid in sids:
        s = Z.subject_entries(sid)
        if s and s["entries"]:
            cache[sid] = s
    return cache


def eval_zeroshot(net, cache, dev):
    net.eval()
    subj_means = []
    for sid, subj in cache.items():
        slices = [(e["img"], e["mask"]) for e in subj["entries"]]
        pred = Z.predict_subject(net, slices, dev)
        d = Z.subject_dice(pred, [e["mask"] for e in subj["entries"]])
        if d:
            subj_means.append(float(np.mean(list(d.values()))))
    a = np.array(subj_means)
    return a


@torch.no_grad()
def adabn_recalibrate(net, cache, dev, bs=16):
    """Reset BN running stats and re-estimate them from the target images (cumulative)."""
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None       # cumulative moving average -> exact target population stats
            m.train()               # update running stats during forward
    # gather all target slices
    allslices = [e["img"] for subj in cache.values() for e in subj["entries"]]
    for i in range(0, len(allslices), bs):
        batch = np.stack(allslices[i:i + bs], 0)
        net(torch.from_numpy(batch).float().to(dev))
    net.eval()
    return len(allslices)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("preprocessing external cohort...", flush=True)
    t0 = time.time()
    cache = build_ext_cache()
    print(f"  {len(cache)} subjects in {time.time()-t0:.1f}s", flush=True)

    net = Z.load_model("unet_r34", dev)

    base = eval_zeroshot(net, cache, dev)
    print(f"[baseline zero-shot] n={len(base)} mean={base.mean():.4f} sd={base.std(ddof=1):.4f} "
          f"range={base.min():.3f}-{base.max():.3f}", flush=True)

    nsl = adabn_recalibrate(net, cache, dev)
    print(f"[AdaBN] recalibrated BN on {nsl} target slices", flush=True)

    adapt = eval_zeroshot(net, cache, dev)
    print(f"[AdaBN adapted]     n={len(adapt)} mean={adapt.mean():.4f} sd={adapt.std(ddof=1):.4f} "
          f"range={adapt.min():.3f}-{adapt.max():.3f}", flush=True)

    out = {"method": "AdaBN source-free adaptation (target images, no labels)",
           "model": "thigh_unet_r34 (4-channel)",
           "baseline_zeroshot": {"n": len(base), "mean": round(float(base.mean()), 4),
                                 "sd": round(float(base.std(ddof=1)), 4),
                                 "min": round(float(base.min()), 3), "max": round(float(base.max()), 3)},
           "adabn": {"n": len(adapt), "mean": round(float(adapt.mean()), 4),
                     "sd": round(float(adapt.std(ddof=1)), 4),
                     "min": round(float(adapt.min()), 3), "max": round(float(adapt.max()), 3)},
           "delta_mean": round(float(adapt.mean() - base.mean()), 4)}
    json.dump(out, open(os.path.join(C.RESULTS, "ext_adabn.json"), "w"), indent=2)
    print(f"wrote results/ext_adabn.json  delta={out['delta_mean']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
