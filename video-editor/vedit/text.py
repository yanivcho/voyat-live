"""Text and compositing.

Hebrew is always shaped by Pillow with the RAQM layout engine (bidi + HarfBuzz), so it reads right-to-left
and never comes out reversed. cv2.putText is used only for ASCII debug labels (contact sheets).
Images here are premultiplied BGRA float32 in [0, 1]: they scale/rotate without dark fringes.
"""
import math
import unicodedata
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

FONTS = Path(__file__).resolve().parent.parent / "fonts"
if not features.check("raqm"):
    raise RuntimeError("Pillow has no RAQM support (Hebrew would render reversed). Install libraqm/libfribidi.")


@lru_cache(maxsize=64)
def font(name, size):
    p = Path(name)
    if not p.exists():
        p = FONTS / (name if p.suffix else name + ".ttf")
    return ImageFont.truetype(str(p), max(1, int(round(size))), layout_engine=ImageFont.Layout.RAQM)


def bidi_class(text):
    """'R' Hebrew/Arabic, 'L' Latin, 'N' numbers or punctuation only."""
    for c in text:
        b = unicodedata.bidirectional(c)
        if b in ("R", "AL"):
            return "R"
        if b == "L":
            return "L"
    return "N"


def text_image(text, fnt, fill=(255, 255, 255), stroke=0, stroke_fill=(0, 0, 0), shadow=None, pad=2):
    """Render one run of text. Returns (img, ox, oy, advance): img is premultiplied BGRA, (ox, oy) is the
    left/baseline origin inside img. shadow = (dx, dy, blur, opacity, rgb) in pixels."""
    rtl = bidi_class(text) == "R"
    kw = dict(direction="rtl" if rtl else "ltr", language="he" if rtl else None)
    asc, desc = fnt.getmetrics()
    adv = fnt.getlength(text, **kw)
    l, t, r, b = fnt.getbbox(text, stroke_width=stroke, anchor="ls", **kw)
    extra = pad + stroke
    if shadow:
        extra += int(abs(shadow[0]) + abs(shadow[1]) + 3 * shadow[2]) + 1
    x0, y0 = min(l, 0), min(t, -asc)
    W = int(math.ceil(max(r, adv) - x0)) + 2 * extra
    H = int(math.ceil(max(b, desc) - y0)) + 2 * extra
    ox, oy = extra - x0, extra - y0

    def cover(width):
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).text((ox, oy), text, font=fnt, fill=255, anchor="ls", stroke_width=width,
                               stroke_fill=255, **kw)
        return np.asarray(m, np.float32) / 255.0

    a_fill = cover(0)
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = np.array(fill[::-1], np.float32) / 255.0
    img = np.dstack([rgb * a_fill[..., None], a_fill])
    if stroke:
        a_st = cover(stroke)
        st = np.dstack([np.array(stroke_fill[::-1], np.float32) / 255.0 * a_st[..., None], a_st])
        img = over(img, st)
    if shadow:
        dx, dy, blur, op, col = shadow
        a = img[..., 3]
        if blur > 0:
            a = cv2.GaussianBlur(a, (0, 0), blur)
        a = np.roll(np.roll(a, int(round(dy)), axis=0), int(round(dx)), axis=1) * op
        sh = np.dstack([np.array(col[::-1], np.float32) / 255.0 * a[..., None], a])
        img = over(img, sh)
    return img, ox, oy, adv


def over(top, bottom):
    """Premultiplied 'top over bottom' (same size)."""
    return top + bottom * (1.0 - top[..., 3:4])


def rounded_rect(w, h, radius, rgb, opacity=1.0, ss=4):
    """Antialiased rounded rectangle as premultiplied BGRA."""
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    r = int(round(min(radius, w / 2, h / 2) * ss))
    m = np.zeros((h * ss, w * ss), np.uint8)
    W, H = w * ss, h * ss
    cv2.rectangle(m, (r, 0), (W - 1 - r, H - 1), 255, -1)
    cv2.rectangle(m, (0, r), (W - 1, H - 1 - r), 255, -1)
    for cx, cy in ((r, r), (W - 1 - r, r), (r, H - 1 - r), (W - 1 - r, H - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
    a = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 * opacity
    return np.dstack([np.array(rgb[::-1], np.float32) / 255.0 * a[..., None], a])


def paste(dst, img, x, y):
    """Premultiplied 'img over dst' for two BGRA float images, img top-left at integer (x, y)."""
    h, w = img.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] = over(img[y0 - y:y1 - y, x0 - x:x1 - x], dst[y0:y1, x0:x1])


def composite(frame, img, cx, cy, scale=1.0, angle=0.0, opacity=1.0):
    """Draw premultiplied BGRA `img` onto a uint8 BGR frame, its center at (cx, cy) (sub-pixel)."""
    if opacity <= 0.002 or scale <= 0.01:
        return
    h, w = img.shape[:2]
    if scale != 1.0 or angle or cx % 1 or cy % 1:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        c, s = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(math.ceil(h * s + w * c)) + 2, int(math.ceil(h * c + w * s)) + 2
        x, y = math.floor(cx - nw / 2), math.floor(cy - nh / 2)
        M[0, 2] += (cx - nw / 2 - x) + nw / 2 - w / 2
        M[1, 2] += (cy - nh / 2 - y) + nh / 2 - h / 2
        img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        x, y = int(cx - w / 2), int(cy - h / 2)
    blit(frame, img, x, y, opacity)


def blit(frame, img, x, y, opacity=1.0):
    H, W = frame.shape[:2]
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = img[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    roi = roi * (1.0 - s[..., 3:4] * opacity) + s[..., :3] * (255.0 * opacity)
    frame[y0:y1, x0:x1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)


def visual_order(classes, widths, space, rtl=True):
    """Left x of each token on a line laid out in base direction RTL (Hebrew) or LTR.
    Consecutive Latin tokens (and numbers that follow Latin) keep their left-to-right order inside a
    right-to-left line, as the Unicode bidi algorithm does."""
    n = len(widths)
    total = sum(widths) + space * max(n - 1, 0)
    xs = [0.0] * n
    if not rtl:
        x = 0.0
        for i in range(n):
            xs[i] = x
            x += widths[i] + space
        return xs, total
    runs, last_strong = [], "R"
    for i, c in enumerate(classes):
        joins = runs and runs[-1][0] == "L" and (c == "L" or (c == "N" and last_strong == "L"))
        if joins:
            runs[-1][1].append(i)
        else:
            runs.append(["L" if c == "L" else c, [i]])
        if c in "RL":
            last_strong = c
    right = total
    for _, idx in runs:
        run_w = sum(widths[i] for i in idx) + space * (len(idx) - 1)
        x = right - run_w
        for i in idx:
            xs[i] = x
            x += widths[i] + space
        right -= run_w + space
    return xs, total
