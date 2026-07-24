"""Strip all embedded metadata (matplotlib 'Software' tag, any text chunks) from figure PNGs.

Run this after (re)generating any figure to keep the submission images free of internal
signatures. Pixels and DPI are preserved; only ancillary text/metadata chunks are removed.

Usage:  python strip_fig_metadata.py
"""
import glob, os
from PIL import Image, PngImagePlugin

LOCS = ["../figures/*.png", "../manuscript_R2/*.png", "../manuscript_reviewed/*.png"]


def strip(f):
    im = Image.open(f)
    had = any(isinstance(v, (str, bytes)) for v in im.info.values())
    dpi = im.info.get("dpi")
    clean = Image.new(im.mode, im.size)
    clean.putdata(list(im.getdata()))
    kw = {"format": "PNG", "pnginfo": PngImagePlugin.PngInfo()}
    if dpi:
        kw["dpi"] = dpi
    clean.save(f, **kw)
    return had


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    n = s = 0
    for pat in LOCS:
        for f in glob.glob(os.path.join(here, pat)):
            n += 1
            if strip(f):
                s += 1
    print(f"checked {n} PNGs, stripped metadata from {s}")


if __name__ == "__main__":
    main()
