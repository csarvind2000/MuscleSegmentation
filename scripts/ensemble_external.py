"""Ensemble + AdaBN + light TTA for label-free external transfer.

Ensembles the domain-randomised models of several architectures (softmax averaging), each
first AdaBN-recalibrated on the external images (source-free) and evaluated with light
flip-based test-time augmentation. No target labels are ever used, and no choice is tuned on
the external Dice. TTA is restricted to horizontal flip because the paper's orientation check
showed this cohort is orientation-sensitive (rotated inputs collapse), so rotational TTA would
hurt rather than help.

Reports each model alone and the ensemble. Writes results/ext_ensemble.json
"""
import os, glob, json, numpy as np, torch, torch.nn as nn
import config as C
import zeroshot_external as Z
from models import build_model
from adabn_external import build_ext_cache

MODELS = ["unet_r34", "unetpp_r34", "deeplabv3p_r34", "segformer_b1", "unet_hrnet18"]


def load_dr(model, dev):
    ck = os.path.join(C.CKPT, f"thigh_{model}_dr.pt")
    net = build_model(model, 4, 14, encoder_weights=None)
    net.load_state_dict(torch.load(ck, map_location="cpu"))
    return net.to(dev).eval()


@torch.no_grad()
def adabn(net, cache, dev, bs=16):
    has_bn = False
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats(); m.momentum = None; m.train(); has_bn = True
    if not has_bn:
        return False
    allslices = [e["img"] for subj in cache.values() for e in subj["entries"]]
    for i in range(0, len(allslices), bs):
        net(torch.from_numpy(np.stack(allslices[i:i + bs], 0)).float().to(dev))
    net.eval()
    return True


@torch.no_grad()
def proba_subject(net, slices, dev, bs=8, tta=True):
    out = []
    for i in range(0, len(slices), bs):
        x = torch.from_numpy(np.stack([s[0] for s in slices[i:i + bs]], 0)).float().to(dev)
        p = torch.softmax(net(x), 1)
        if tta:                                   # horizontal-flip TTA only (safe orientation)
            pf = torch.softmax(net(torch.flip(x, dims=[3])), 1)
            p = 0.5 * (p + torch.flip(pf, dims=[3]))
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)                 # [N, nc, H, W]


def summ(subj_means):
    a = np.array(subj_means)
    return {"n": len(a), "mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = build_ext_cache()
    print(f"external subjects: {len(cache)}", flush=True)

    nets = {}
    for m in MODELS:
        if not os.path.exists(os.path.join(C.CKPT, f"thigh_{m}_dr.pt")):
            print(f"  (skip {m}: no DR checkpoint)"); continue
        net = load_dr(m, dev)
        used_bn = adabn(net, cache, dev)
        nets[m] = net
        print(f"  loaded {m} (AdaBN={'yes' if used_bn else 'n/a'})", flush=True)

    per_model = {m: [] for m in nets}
    ens_all, ens_bn = [], []
    bn_models = [m for m in nets if m != "segformer_b1"]     # resnet-based subset

    for sid, subj in cache.items():
        slices = [(e["img"], e["mask"]) for e in subj["entries"]]
        gts = [e["mask"] for e in subj["entries"]]
        probs = {m: proba_subject(net, slices, dev) for m, net in nets.items()}
        for m, pr in probs.items():
            d = Z.subject_dice(pr.argmax(1), gts)
            if d: per_model[m].append(float(np.mean(list(d.values()))))
        # full ensemble (all architectures)
        ens = sum(probs.values())
        d = Z.subject_dice(ens.argmax(1), gts)
        if d: ens_all.append(float(np.mean(list(d.values()))))
        # resnet-only ensemble
        if bn_models:
            ensb = sum(probs[m] for m in bn_models)
            d = Z.subject_dice(ensb.argmax(1), gts)
            if d: ens_bn.append(float(np.mean(list(d.values()))))

    out = {"note": "DR + AdaBN + hflip-TTA per model; ensemble = softmax average. Label-free "
                   "(target images, no labels); no selection tuned on external Dice.",
           "per_model_dr_adabn_tta": {m: summ(v) for m, v in per_model.items()},
           "ensemble_all": summ(ens_all),
           "ensemble_resnet_only": summ(ens_bn) if ens_bn else None}
    json.dump(out, open(os.path.join(C.RESULTS, "ext_ensemble.json"), "w"), indent=2)
    print("\n=== per model (DR+AdaBN+TTA) ===")
    for m, v in out["per_model_dr_adabn_tta"].items():
        print(f"  {m:16s} mean={v['mean']:.4f} sd={v['sd']:.4f}")
    print(f"=== ensemble (all {len(nets)}) mean={out['ensemble_all']['mean']:.4f} "
          f"sd={out['ensemble_all']['sd']:.4f} ===")
    if out["ensemble_resnet_only"]:
        print(f"=== ensemble (resnet-only) mean={out['ensemble_resnet_only']['mean']:.4f} ===")
    print("wrote results/ext_ensemble.json", flush=True)


if __name__ == "__main__":
    main()
