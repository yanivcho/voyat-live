"""Picture treatments and person-aware layers."""
import cv2
import numpy as np

from .captions import Layer


class Grade(Layer):
    """Color grade: contrast S-curve, saturation, warmth, teal/orange split, vignette."""
    z = 0

    def __init__(self, contrast=0.0, saturation=1.0, warmth=0.0, teal_orange=0.0, lift=0.0, vignette=0.0):
        self.kw = dict(contrast=contrast, saturation=saturation, warmth=warmth, teal_orange=teal_orange,
                       lift=lift, vignette=vignette)

    def prepare(self, p):
        k = self.kw
        x = np.arange(256, dtype=np.float32) / 255.0
        y = k["lift"] + (1 - k["lift"]) * x
        y = y + k["contrast"] * (y * y * (3 - 2 * y) - y)
        lo, hi = (1 - y) ** 2, y ** 2  # shadow / highlight weights
        to, w = k["teal_orange"], k["warmth"]
        b = y + to * (0.06 * lo - 0.05 * hi) - 0.04 * w * y
        g = y + to * 0.02 * lo
        r = y + to * (-0.05 * lo + 0.06 * hi) + 0.04 * w * y
        self.lut = (np.clip(np.stack([b, g, r], -1), 0, 1) * 255 + 0.5).astype(np.uint8).reshape(1, 256, 3)
        self.vig = None
        if k["vignette"]:
            W, H = p.size
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            d = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / np.sqrt(2)
            self.vig = (1 - k["vignette"] * np.clip((d - 0.35) / 0.65, 0, 1) ** 2)[..., None]

    def render(self, frame, ctx):
        out = cv2.LUT(frame, self.lut)
        s = self.kw["saturation"]
        if s != 1:
            gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(out, s, gray, 1 - s, 0)
        if self.vig is not None:
            out = (out * self.vig + 0.5).astype(np.uint8)
        frame[:] = out


class Letterbox(Layer):
    """Cinematic bars (e.g. 2.39:1) that can slide in."""
    z = 15

    def __init__(self, ratio=2.39, slide_in=0.0, color=(0, 0, 0)):
        self.ratio, self.slide, self.color = ratio, slide_in, color[::-1]

    def prepare(self, p):
        W, H = p.size
        self.bar = max(0, int(round((H - W / self.ratio) / 2)))

    def render(self, frame, ctx):
        k = min(1.0, ctx.t / self.slide) if self.slide else 1.0
        h = int(round(self.bar * (1 - (1 - k) ** 3)))
        if h > 0:
            frame[:h], frame[frame.shape[0] - h:] = self.color, self.color


class BehindPerson(Layer):
    """Draw another layer BEHIND the person (text between the background and you). Needs the mask."""

    def __init__(self, layer):
        self.layer, self.z = layer, layer.z

    def prepare(self, p):
        self.layer.prepare(p)

    def render(self, frame, ctx):
        before = frame.copy()
        self.layer.render(frame, ctx)
        a = ctx.person()[..., None]
        frame[:] = (before * a + frame * (1.0 - a) + 0.5).astype(np.uint8)


class ReplaceBackground(Layer):
    """Keep the person, replace everything else with a color (BGR tuple) or an image path."""
    z = 1

    def __init__(self, background, blur=0):
        self.bg_spec, self.blur = background, blur

    def prepare(self, p):
        W, H = p.size
        if isinstance(self.bg_spec, (tuple, list)):
            self.bg = np.full((H, W, 3), self.bg_spec[::-1], np.uint8)
        else:
            img = cv2.imread(str(self.bg_spec))
            s = max(W / img.shape[1], H / img.shape[0])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            y, x = (img.shape[0] - H) // 2, (img.shape[1] - W) // 2
            self.bg = img[y:y + H, x:x + W]
        if self.blur:
            self.bg = cv2.GaussianBlur(self.bg, (0, 0), self.blur)

    def render(self, frame, ctx):
        a = ctx.person()[..., None]
        frame[:] = (frame * a + self.bg * (1.0 - a) + 0.5).astype(np.uint8)
