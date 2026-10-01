"""Animated word-by-word captions. Every word is anchored to its SOURCE time and mapped through the time map."""
import bisect
import itertools
import math

import cv2
import numpy as np

from .text import bidi_class, composite, font, paste, rounded_rect, text_image, visual_order


BOX_PAD = 0.10  # highlight box padding around the spoken word, fraction of font size


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_cubic(x):
    return 1 - (1 - x) ** 3


def ease_out_back(x, c1=2.2):
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


class Layer:
    z = 10  # draw order: 0 = picture treatments, 10 = overlays, 20 = captions

    def prepare(self, p):
        """Called once before rendering, after the whole edit (cuts, freezes) is defined."""

    def render(self, frame, ctx):
        raise NotImplementedError


class Captions(Layer):
    z = 20

    def __init__(self, style, words=None, pages=None, y=None):
        """words: [{'text', 'start', 'end'}] in source seconds (None = the transcript, auto-paginated).
        pages: explicit grouping [[word, ...], ...], e.g. on-screen text you wrote yourself."""
        self.st, self._words, self._pages, self._y = style, words, pages, y

    def _clean(self, text):
        return text.strip().strip(self.st.strip).strip()

    def prepare(self, p):
        st, tm = self.st, p.timemap
        self.W, self.H = p.size
        self.size = st.size * self.W
        self.fnt = font(st.font, self.size)
        self.cap = -self.fnt.getbbox("שלום", anchor="ls")[1]  # height of Hebrew letters above the baseline
        self.cy = (self._y if self._y is not None else st.y.get(p.format_name, 0.75)) * self.H

        def out(w):
            t0 = tm.to_out(w["start"], "next")
            t1 = tm.to_out(w["end"], "prev")
            return dict(raw=w["text"], text=self._clean(w["text"]), t0=t0, t1=max(t1, t0 + 0.05))

        if self._pages is None:
            ws = [out(w) for w in (self._words if self._words is not None else p.words) if not w.get("flag")]
            pages = self._paginate([w for w in ws if w["text"]])
        else:
            pages = [[out(w) for w in pg if self._clean(w["text"])] for pg in self._pages]
        self.pages = [self._layout(pg) for pg in pages if pg]
        self._timing()
        self._cache = {}

    def _paginate(self, ws):
        st, pages, cur = self.st, [], []
        for w in ws:
            if cur:
                chars = sum(len(x["text"]) + 1 for x in cur) + len(w["text"])
                pause = w["t0"] - cur[-1]["t1"]
                if (len(cur) >= st.max_words or chars > st.max_chars or pause > st.gap_break
                        or (pause >= st.soft_break and len(cur) >= 2) or cur[-1]["raw"].rstrip()[-1:] in ".?!"):
                    pages.append(cur)
                    cur = []
            cur.append(w)
        return pages + ([cur] if cur else [])

    def _layout(self, ws):
        st, f, size = self.st, self.fnt, self.size
        stroke = int(round(st.stroke * size))
        sh = (st.shadow[0] * size, st.shadow[1] * size, st.shadow[2] * size, st.shadow[3], (0, 0, 0)) if st.shadow else None
        for w in ws:
            w["img"] = text_image(w["text"], f, st.fill, stroke, st.stroke_fill, sh)
            if st.active_box:
                w["img_on"] = text_image(w["text"], f, st.active)
            elif st.active != st.fill:
                w["img_on"] = text_image(w["text"], f, st.active, stroke, st.stroke_fill, sh)
            else:
                w["img_on"] = w["img"]
            w["cls"] = bidi_class(w["text"])
        # word gap: a normal space plus room for outlines and the highlight box, so neighbours never touch
        space = f.getlength(" ") + 2 * stroke + (2 * BOX_PAD * size if st.active_box else 0)
        lines, cur, cw = [], [], 0.0
        for w in ws:
            adv = w["img"][3]
            if cur and cw + space + adv > st.max_width * self.W:
                lines.append(cur)
                cur, cw = [], 0.0
            cw += (space if cur else 0.0) + adv
            cur.append(w)
        lines.append(cur)
        if len(lines) > 1:  # balance the lines (no lonely last word): minimize the widest line
            adv = [w["img"][3] for w in ws]
            best = None
            for cuts in itertools.combinations(range(1, len(ws)), len(lines) - 1):
                b = (0, *cuts, len(ws))
                widest = max(sum(adv[i:j]) + space * (j - i - 1) for i, j in zip(b, b[1:]))
                if widest <= st.max_width * self.W and (best is None or widest < best[0]):
                    best = (widest, b)
            if best:
                lines = [ws[i:j] for i, j in zip(best[1], best[1][1:])]
        rtl = sum(w["cls"] == "R" for w in ws) >= sum(w["cls"] == "L" for w in ws)
        pill_h = self.cap + 0.62 * size
        lh = max(st.line_spacing * size, pill_h + 0.14 * size if st.pill else 0)
        widths = []
        for li, line in enumerate(lines):
            xs, total = visual_order([w["cls"] for w in line], [w["img"][3] for w in line], space, rtl)
            widths.append(total)
            for w, x in zip(line, xs):
                w["x"] = x - total / 2                                       # left of advance box, from center
                w["base"] = (li - (len(lines) - 1) / 2) * lh + self.cap / 2  # baseline, from block center
        margin = int(0.6 * size + 4 * (stroke + (abs(sh[1]) + 3 * sh[2] if sh else 0)))
        pw = int(max(widths) + 2 * margin) | 1
        ph = int(len(lines) * lh + 2 * margin) | 1
        return dict(words=ws, lines=lines, widths=widths, lh=lh, pill_h=pill_h, pw=pw, ph=ph)

    def _timing(self):
        st = self.st
        for i, pg in enumerate(self.pages):
            start = pg["words"][0]["t0"] - st.lead
            if i:
                start = max(start, self.pages[i - 1]["words"][-1]["t1"])
            pg["start"] = max(0.0, start)
        for i, pg in enumerate(self.pages):
            natural = pg["words"][-1]["t1"] + st.hold
            nxt = self.pages[i + 1]["start"] if i + 1 < len(self.pages) else math.inf
            pg["end"], pg["fade_out"] = min(natural, nxt), natural < nxt
            ws = pg["words"]
            for j, w in enumerate(ws):
                w["until"] = ws[j + 1]["t0"] if j + 1 < len(ws) else pg["end"]
        self._starts = [pg["start"] for pg in self.pages]

    def _compose(self, k, active):
        key = (k, active)
        if key in self._cache:
            return self._cache[key]
        st, pg, size = self.st, self.pages[k], self.size
        cx, cy = pg["pw"] / 2, pg["ph"] / 2
        img = np.zeros((pg["ph"], pg["pw"], 4), np.float32)
        if st.pill:
            for li, w_line in enumerate(pg["widths"]):
                box = rounded_rect(w_line + 0.76 * size, pg["pill_h"], 0.3 * size, st.pill[0], st.pill[1])
                yc = cy + (li - (len(pg["widths"]) - 1) / 2) * pg["lh"]
                paste(img, box, int(round(cx - box.shape[1] / 2)), int(round(yc - box.shape[0] / 2)))
        for j, w in enumerate(pg["words"]):
            on = j == active
            sprite, ox, oy, adv = w["img_on"] if on else w["img"]
            upcoming = active is None or j > active
            if upcoming and st.upcoming_alpha < 1:
                sprite = sprite * st.upcoming_alpha
            x, base = cx + w["x"], cy + w["base"]
            if on and st.active_box:
                box = rounded_rect(adv + 2 * BOX_PAD * size, self.cap + 0.36 * size, 0.16 * size, st.active_box)
                bx, by = int(round(x + adv / 2 - box.shape[1] / 2)), int(round(base - self.cap / 2 - box.shape[0] / 2))
                part = box.copy()
                paste(part, sprite, int(round(x - ox)) - bx, int(round(base - oy)) - by)
                if st.active_scale != 1:
                    part = cv2.resize(part, None, fx=st.active_scale, fy=st.active_scale, interpolation=cv2.INTER_LINEAR)
                paste(img, part, int(round(x + adv / 2 - part.shape[1] / 2)), int(round(base - self.cap / 2 - part.shape[0] / 2)))
            else:
                paste(img, sprite, int(round(x - ox)), int(round(base - oy)))
        self._cache[key] = img
        return img

    def render(self, frame, ctx):
        t, st = ctx.t, self.st
        k = bisect.bisect_right(self._starts, t) - 1
        if k < 0 or t >= self.pages[k]["end"]:
            return
        pg = self.pages[k]
        active = next((j for j, w in enumerate(pg["words"]) if w["t0"] <= t < w["until"]), None)
        e = clamp01((t - pg["start"]) / st.enter) if st.enter else 1.0
        fade = clamp01((pg["end"] - t) / st.exit) if st.exit and pg["fade_out"] else 1.0
        dy, scale = 0.0, 1.0
        if st.anim == "pop":
            scale, opacity = 0.55 + 0.45 * ease_out_back(e), min(1.0, 4 * e) * fade
        else:
            ee = ease_out_cubic(e)
            opacity, dy = ee * fade, (1 - ee) * 0.25 * self.size
        composite(frame, self._compose(k, active), self.W / 2, self.cy + dy, scale=scale, opacity=opacity)
