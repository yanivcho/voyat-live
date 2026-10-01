"""Person segmentation: one rembg u2net_human_seg mask per frame, computed once and cached in work/masks.npy."""
import json
import os
import time
from pathlib import Path

import numpy as np

from .media import FrameReader


def compute(work, long_side=640, model="u2net_human_seg", log=print):
    work = Path(work)
    info = json.loads((work / "info.json").read_text())
    W, H, n = info["width"], info["height"], info["frames"]
    s = min(1.0, long_side / max(W, H))
    mw, mh = int(round(W * s / 2)) * 2, int(round(H * s / 2)) * 2
    from PIL import Image
    from rembg import new_session
    sess = new_session(model)
    reader = FrameReader(work / "master.mp4", W, H, info["fps"], n, size=(mw, mh))
    raw = np.lib.format.open_memmap(work / "masks.raw.npy", mode="w+", dtype=np.uint8, shape=(n, mh, mw))
    t0 = time.time()
    for i in range(n):
        raw[i] = np.asarray(sess.predict(Image.fromarray(np.ascontiguousarray(reader.get(i)[:, :, ::-1])))[0])
        if log and (i % 50 == 0 or i == n - 1):
            log(f"mask {i + 1}/{n}  ({(time.time() - t0) / (i + 1):.2f}s/frame)")
    reader.close()
    # light symmetric temporal smoothing (1-2-1): removes flicker without lagging behind motion
    out = np.lib.format.open_memmap(work / "masks.tmp.npy", mode="w+", dtype=np.uint8, shape=(n, mh, mw))
    for i in range(n):
        a, b, c = raw[max(i - 1, 0)], raw[i], raw[min(i + 1, n - 1)]
        out[i] = ((a.astype(np.uint16) + 2 * b + c + 2) // 4).astype(np.uint8)
    out.flush()
    del out, raw
    os.replace(work / "masks.tmp.npy", work / "masks.npy")
    os.remove(work / "masks.raw.npy")
    return work / "masks.npy"


def load(work):
    p = Path(work) / "masks.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None
