"""Side-by-side acquisition comparison of the primary (training) and external cohorts,
to document the cross-cohort domain gap visually (reviewer R1). For each cohort we show an
axial slice (in-plane) and a coronal reformat resampled to true physical proportions, so the
external cohort's 26 mm / 14-slice anisotropy appears as coarse bands. Writes
figures/fig_cohort_compare_R1.png(.pdf).
"""
import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

EXT = ("/media/ranjhaa-local/volume23/sarcopenia/datasets/opensource_dataset/"
       "multiethnic_thigh/HuashanMyo/HuashanMyo/Water/THIGH_002_0001.nii.gz")
PRIM = ("/media/ranjhaa-local/volume23/sarcopenia/datasets/"
        "TTSH_Thigh_sarcopenia_2012_dataset/thigh_input_4_contrast/Subject14_wateronly.nii.gz")


def load(f):
    im = nib.as_closest_canonical(nib.load(f))
    return im.get_fdata().astype(np.float32), tuple(float(z) for z in im.header.get_zooms()[:3])


def norm(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def axial(vol):
    return norm(vol[:, :, vol.shape[2] // 2]).T          # X-Y at mid Z


def coronal_phys(vol, sp):
    """Coronal reformat (X-Z at mid Y) resampled to true mm proportions."""
    sl = vol[:, vol.shape[1] // 2, :]                    # (X, Z)
    wmm = sl.shape[0] * sp[0]                            # physical width  (in-plane)
    hmm = sl.shape[1] * sp[2]                            # physical height (through-plane)
    px = 0.8                                             # mm per output pixel
    out = resize(sl, (int(round(hmm / px)), int(round(wmm / px))), order=0,
                 preserve_range=True, anti_aliasing=False)  # order=0 keeps the blocky slabs
    return norm(out[::-1])                               # flip so superior is up


def main():
    ev, es = load(EXT); pv, ps = load(PRIM)
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 8.4))
    panels = [
        (0, 0, axial(ev),            f"External (HuashanMyo)\naxial in-plane view"),
        (0, 1, coronal_phys(ev, es), f"External coronal reformat\n14 slices, 26 mm through-plane"),
        (1, 0, axial(pv),            f"Primary (Dixon thigh, training)\naxial in-plane view"),
        (1, 1, coronal_phys(pv, ps), f"Primary coronal reformat\n$\\approx$60 slices, 5 mm through-plane"),
    ]
    for r, c, img, title in panels:
        ax[r, c].imshow(img, cmap="gray", aspect="equal")
        ax[r, c].set_title(title, fontsize=10.5)
        ax[r, c].axis("off")
    fig.suptitle("Acquisition mismatch between the training and external cohorts",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        md = {"Software": None} if ext == "png" else {"Creator": None, "Producer": None}
        fig.savefig(os.path.join(C.FIG, f"fig_cohort_compare_R1.{ext}"), dpi=170,
                    bbox_inches="tight", metadata=md)
    plt.close()
    print(f"wrote fig_cohort_compare_R1  ext{ev.shape}{es}  prim{pv.shape}{ps}")


if __name__ == "__main__":
    main()
