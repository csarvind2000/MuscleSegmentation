"""Post-hoc analyses (reviewer items 7, 8):
  - MTL efficiency: parameters, checkpoint size, GPU memory, inference time for
    two single-task models vs one shared multi-task model.
  - Gate functional importance: on the trained DixonFuse model, zero each input
    contrast in turn and measure the test-Dice drop (occlusion), complementing the
    average gate weights.
Writes results/efficiency.json and results/gate_occlusion.json.
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import config as C
from datasets import ThighSliceDataset
from models_method import DixonFuseUNet
from train import evaluate
from train_method import subject_files

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def efficiency():
    a_nc, t_nc = C.AATTCT_NUM_CLASSES, 14
    ct = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=a_nc)
    mri = DixonFuseUNet("dixonfuse", t_nc)
    from train_mtl_norm import SharedMTL
    mtl = SharedMTL("shared_bn", a_nc, t_nc, 4)
    p_ct, p_mri, p_mtl = nparams(ct), nparams(mri), nparams(mtl)

    def ckpt_mb(m):
        path = os.path.join(C.CKPT, "_tmp.pt"); torch.save(m.state_dict(), path)
        mb = os.path.getsize(path) / 1e6; os.remove(path); return round(mb, 1)

    # inference time + peak memory (MRI branch, 256x256)
    def timeit(fn, ch):
        if DEV != "cuda":
            return None, None
        x = torch.randn(1, ch, 256, 256, device=DEV)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        for _ in range(3): fn(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20): fn(x)
        torch.cuda.synchronize(); dt = (time.time() - t) / 20 * 1000
        mem = torch.cuda.max_memory_allocated() / 1e6
        return round(dt, 1), round(mem, 0)

    res = {
        "params_M": {"single_ct": round(p_ct / 1e6, 2), "single_mri_dixonfuse": round(p_mri / 1e6, 2),
                     "single_total": round((p_ct + p_mri) / 1e6, 2), "multitask_shared": round(p_mtl / 1e6, 2)},
        "checkpoint_MB": {"single_total": ckpt_mb(ct) + ckpt_mb(mri), "multitask_shared": ckpt_mb(mtl)},
    }
    if DEV == "cuda":
        mri = mri.to(DEV).eval(); mtl = mtl.to(DEV).eval()
        with torch.no_grad():
            dt_s, mem_s = timeit(lambda x: mri(torch.cat([x, x[:, :1]], 1)), 5)  # dixonfuse needs 5ch
            dt_m, mem_m = timeit(lambda x: mtl.mri(x), 4)
        res["inference_ms"] = {"single_mri": dt_s, "multitask_mri": dt_m}
        res["peak_mem_MB"] = {"single_mri": mem_s, "multitask_mri": mem_m}
    json.dump(res, open(os.path.join(C.RESULTS, "efficiency.json"), "w"), indent=2)
    print("efficiency:", json.dumps(res))


@torch.no_grad()
def gate_occlusion():
    meta = json.load(open(os.path.join(C.CACHE, "thigh5_meta.json")))
    nc = meta["num_classes"]; names = meta["channel_names"]
    cache5 = os.path.join(C.CACHE, "thigh_slices5")
    te = DataLoader(ThighSliceDataset(subject_files(meta["subjects_test"], cache5)), 8, False, num_workers=6, pin_memory=True)
    model = DixonFuseUNet("dixonfuse", nc).to(DEV)
    model.load_state_dict(torch.load(os.path.join(C.CKPT, "method_dixonfuse_n18_s0.pt"), map_location=DEV))
    model.eval()

    class Occ(nn.Module):
        def __init__(s, ch): super().__init__(); s.ch = ch
        def forward(s, x):
            if s.ch is not None:
                x = x.clone(); x[:, s.ch] = 0
            return model(x)

    base, _ = evaluate(Occ(None).to(DEV), te, nc, DEV)
    base = float(np.nanmean(base[1:]))
    drops = {}
    for ci in range(5):
        d, _ = evaluate(Occ(ci).to(DEV), te, nc, DEV)
        drops[names[ci]] = round(base - float(np.nanmean(d[1:])), 4)
    out = {"baseline_dice": round(base, 4), "dice_drop_when_channel_zeroed": drops}
    json.dump(out, open(os.path.join(C.RESULTS, "gate_occlusion.json"), "w"), indent=2)
    print("occlusion:", json.dumps(out))


if __name__ == "__main__":
    efficiency()
    gate_occlusion()
