"""Self-supervised encoder pretraining on unlabelled Dixon MRI (reviewer item 5).

A masked-reconstruction (denoising autoencoder) objective: random square patches of
the 4-channel Dixon input are zeroed and the network reconstructs the full input.
Only the non-test subjects are used, so the downstream evaluation stays leakage-free.
The pretrained ResNet-34 encoder is saved for use as a segmentation initialisation
(init 'ssl' in train_lowdata.py), giving a medical, in-modality alternative to
ImageNet.

Usage: python pretrain_ssl.py --region thigh --epochs 150
Writes checkpoints/ssl_encoder_<region>.pt
"""
import os, glob, json, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import config as C


class UnlabelledDixon(Dataset):
    def __init__(self, files, mask_frac=0.5, patch=32):
        self.files = files; self.mask_frac = mask_frac; self.patch = patch
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = np.load(self.files[i])["img"][:4].astype(np.float32)  # 4xHxW
        C_, H, W = img.shape
        corrupt = img.copy()
        n = int(self.mask_frac * (H * W) / (self.patch ** 2))
        for _ in range(max(1, n)):
            y = np.random.randint(0, H - self.patch); x = np.random.randint(0, W - self.patch)
            corrupt[:, y:y+self.patch, x:x+self.patch] = 0.0
        return torch.from_numpy(corrupt), torch.from_numpy(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="thigh", choices=["thigh", "calf"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cache = os.path.join(C.CACHE, f"{args.region}_slices5")
    meta = json.load(open(os.path.join(C.CACHE, f"{args.region}5_meta.json")))
    subs = meta["subjects_train"] + meta["subjects_val"]   # exclude test subjects
    files = [f for s in subs for f in sorted(glob.glob(os.path.join(cache, f"{s}_*.npz")))]
    dl = DataLoader(UnlabelledDixon(files), args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)
    dev = "cuda"
    net = smp.Unet("resnet34", encoder_weights=None, in_channels=4, classes=4).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    print(f"SSL pretraining on {len(subs)} subjects, {len(files)} slices", flush=True)
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
    path = os.path.join(C.CKPT, f"ssl_encoder_{args.region}.pt")
    torch.save(net.encoder.state_dict(), path)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
