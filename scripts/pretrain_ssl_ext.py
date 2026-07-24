"""Self-supervised pretraining on the UNLABELLED external (HuashanMyo) images (R1.1).

Tests whether SSL on the target cohort's own images closes the cross-cohort domain gap
with fewer target labels. Masked-reconstruction (denoising autoencoder) on the 3-channel
[Water, Fat, FF] external slices, using ONLY the 56 non-test subjects (train+val) so the
fixed 10-subject test set stays completely unseen (leakage-free). The pretrained
ResNet-34 encoder is saved for use as init='ssl' in train_external.py.

Usage: python pretrain_ssl_ext.py --epochs 150
Writes checkpoints/ssl_encoder_ext_wff.pt   (3-channel encoder)
"""
import os, glob, json, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import config as C

CACHE = os.path.join(C.CACHE, "ext_slices3")


class UnlabelledExt(Dataset):
    def __init__(self, files, mask_frac=0.5, patch=32):
        self.files = files; self.mask_frac = mask_frac; self.patch = patch
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = np.load(self.files[i])["img"][:3].astype(np.float32)   # 3xHxW (W,F,FF)
        _, H, W = img.shape
        corrupt = img.copy()
        n = int(self.mask_frac * (H * W) / (self.patch ** 2))
        for _ in range(max(1, n)):
            y = np.random.randint(0, H - self.patch); x = np.random.randint(0, W - self.patch)
            corrupt[:, y:y+self.patch, x:x+self.patch] = 0.0
        return torch.from_numpy(corrupt), torch.from_numpy(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(C.CACHE, "ext_meta.json")))
    subs = meta["subjects_train"] + meta["subjects_val"]            # exclude the 10 test subjects
    files = [f for s in subs for f in sorted(glob.glob(os.path.join(CACHE, f"{s}_*.npz")))]
    dl = DataLoader(UnlabelledExt(files), args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)
    dev = "cuda"
    net = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=3).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    print(f"SSL(ext) on {len(subs)} target subjects, {len(files)} unlabelled slices", flush=True)
    for ep in range(args.epochs):
        net.train(); run = 0.0
        for corrupt, target in dl:
            corrupt, target = corrupt.to(dev), target.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = nn.functional.mse_loss(net(corrupt), target)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        if (ep + 1) % 25 == 0:
            print(f"ep{ep+1}/{args.epochs} mse={run/len(dl):.4f}", flush=True)
    path = os.path.join(C.CKPT, "ssl_encoder_ext_wff.pt")
    torch.save(net.encoder.state_dict(), path)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
