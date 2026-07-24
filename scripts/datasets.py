"""Torch datasets for the two benchmark tasks (both cast to 2D multi-class)."""
import os, csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# ---------------- AATTCT (abdominal CT adipose) ----------------
class AATTCTDataset(Dataset):
    """Reads jpg CT + SAT/VAT png masks, fuses to 3-class label. Grayscale in."""
    def __init__(self, index_csv, split, img_size=512, augment=False):
        self.rows = []
        with open(index_csv) as f:
            for r in csv.DictReader(f):
                if r["split"] == split:
                    self.rows.append(r)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = np.array(Image.open(r["image"]).convert("L"), dtype=np.float32)
        sat = (np.array(Image.open(r["sat"]).convert("L")) > 127)
        vat = (np.array(Image.open(r["vat"]).convert("L")) > 127)
        mask = np.zeros(img.shape, dtype=np.int64)
        mask[sat] = 1
        mask[vat] = 2
        img = img / 255.0
        if self.augment:
            img, mask = _aug2d(img, mask)
        img = torch.from_numpy(img)[None].float()           # 1xHxW
        mask = torch.from_numpy(mask).long()
        return img, mask


# ---------------- Thigh MRI muscle (cached 2D slices) ----------------
class ThighSliceDataset(Dataset):
    """Reads cached .npz slices: arr 'img' (CxHxW float32), 'mask' (HxW int64)."""
    def __init__(self, file_list, augment=False):
        self.files = file_list
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = d["img"].astype(np.float32)      # CxHxW, already z-normed per volume
        mask = d["mask"].astype(np.int64)      # HxW
        if self.augment:
            img, mask = _aug2d_multi(img, mask)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


# ---------------- light augmentation (no external dep required) ----------------
def _aug2d(img, mask):
    if np.random.rand() < 0.5:
        img = img[:, ::-1].copy(); mask = mask[:, ::-1].copy()
    k = np.random.randint(0, 4)
    if k:
        img = np.rot90(img, k).copy(); mask = np.rot90(mask, k).copy()
    return img, mask

def _aug2d_multi(img, mask):
    if np.random.rand() < 0.5:
        img = img[:, :, ::-1].copy(); mask = mask[:, ::-1].copy()
    k = np.random.randint(0, 4)
    if k:
        img = np.rot90(img, k, axes=(1, 2)).copy(); mask = np.rot90(mask, k).copy()
    return img, mask
