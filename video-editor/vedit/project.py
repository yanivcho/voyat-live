"""Project = the edit of one video: timeline + layers. `frame_at(t)` renders any output moment;
`render()` renders the whole movie on all cores and muxes the sound cut with the same time map."""
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import time
import unicodedata
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import audio, contact, mask
from .captions import Captions, clamp01
from .media import BT709, FFMPEG, SWS, FrameReader, run
from .styles import FORMATS, STYLES
from .timemap import Segment, TimeMap, silence_cut


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_in_out(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def norm(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.category(c).startswith(("M", "P"))).strip().lower()


def warp(src, M, size, dst=None):
    """Affine-place src on a canvas: INTER_AREA prefilter for strong downscales, Lanczos for upscales."""
    s = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
    if s < 0.75:
        h, w = src.shape[:2]
        sw, sh = max(1, round(w * s)), max(1, round(h * s))
        src = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        M = M.copy()
        M[:, 0] *= w / sw
        M[:, 1] *= h / sh
        s = 1.0
    flags = cv2.INTER_LANCZOS4 if s > 1.25 else cv2.INTER_LINEAR
    if dst is not None:
        return cv2.warpAffine(src, M, size, dst=dst, flags=flags, borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.warpAffine(src, M, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)


class Ctx:
    """What a layer knows about the frame being drawn."""

    def __init__(self, p, t, srcs):
        self.p, self.t, self.srcs, self._person = p, t, srcs, None

    def person(self):
        """Person alpha (H x W float 0..1) in canvas coordinates."""
        if self._person is None:
            self._person = self.p._person_alpha(self.t, self.srcs)
        return self._person


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.work, self.out = self.root / "work", self.root / "out"
        self.info = json.loads((self.work / "info.json").read_text())
        self.fps = Fraction(self.info["fps"])
        self.src_w, self.src_h, self.frames = self.info["width"], self.info["height"], self.info["frames"]
        self.src_duration = self.info["duration"]
        wj = self.work / "words.json"
        self.words = json.loads(wj.read_text())["words"] if wj.exists() else []
        self.timemap = TimeMap.identity(self.src_duration)
        self.layers, self.zooms, self.sounds, self.report = [], [], [], {}
        self.fit, self.focus, self.sharpen, self.loudness = "cover", (0.5, 0.5), "auto", -14.0
        self.set_format("vertical")
        self.set_style("clean")
        self._pid = self._readers = self._masks = None
        self._prepared = False

    # ---------- format & look ----------
    def set_format(self, name="vertical", fit=None, focus=None):
        """name: vertical | horizontal | square. fit: cover (fill, crop) | contain (whole frame + blurred fill)."""
        self.format_name, self.size = name, FORMATS[name]
        self.fit = fit or self.fit
        self.focus = focus or self.focus

    def set_style(self, name, **overrides):
        self.style = replace(STYLES[name], **overrides)

    # ---------- anchors: int = word number, str = word text, float = source seconds ----------
    def find(self, text, occurrence=1):
        key, n = norm(text), 0
        for i, w in enumerate(self.words):
            if norm(w["text"]) == key:
                n += 1
                if n == occurrence:
                    return i
        raise KeyError(f"word {text!r} #{occurrence} is not in the transcript")

    def src_time(self, at, edge="start"):
        if isinstance(at, str):
            at = self.find(at)
        if isinstance(at, int):
            return self.words[at][edge]
        return float(at)

    def out_time(self, at, edge="start"):
        return self.timemap.to_out(self.src_time(at, edge), "next" if edge == "start" else "prev")

    # ---------- timeline ----------
    def cut_silences(self, **kw):
        """Remove pauses (speech found by Silero VAD + the transcript). kw: min_silence, keep_gap, xfade, lead, tail."""
        wav = self.work / "audio16k.wav"
        regions = audio.speech_regions(audio.load_wav(wav)[0][:, 0], self.words)[0] if wav.exists() else []
        self.timemap, rep = silence_cut(self.src_duration, regions, **kw)
        rep["speech"] = [(round(a, 2), round(b, 2)) for a, b in regions]
        self.report["silence_cut"] = rep
        return rep

    def keep(self, ranges, xfade=0.0):
        """Manual trim: keep only these source ranges [(start, end), ...]."""
        self.timemap = TimeMap([Segment(a, b, xfade=xfade if i else 0.0) for i, (a, b) in enumerate(ranges)])

    def freeze(self, at, hold):
        self.timemap = self.timemap.freeze(self.src_time(at), hold)

    # ---------- content ----------
    def add(self, layer):
        self.layers.append(layer)
        return layer

    def captions(self, words=None, y=None, **style):
        return self.add(Captions(replace(self.style, **style), words=words, y=y))

    def text(self, text, at, until=None, per_word=0.15, y=None, **style):
        """On-screen text you wrote (not from the transcript); its words appear one by one from `at`."""
        t0 = self.src_time(at)
        ws = [dict(text=tok, start=t0 + i * per_word, end=t0 + (i + 1) * per_word) for i, tok in enumerate(text.split())]
        if until is not None:
            ws[-1]["end"] = max(ws[-1]["end"], self.src_time(until, "end"))
            style.setdefault("hold", 0.0)
        return self.add(Captions(replace(self.style, **style), pages=[ws], y=y))

    def zoom(self, at, scale=1.15, until=None, attack=0.15, release=0.3, center=(0.5, 0.45)):
        """Punch-in (short attack) or slow push-in (long attack) from `at` until `until` (None = to the end)."""
        self.zooms.append(dict(at=at, until=until, scale=scale, attack=attack, release=release, center=center))

    def sound(self, path, at, gain_db=0.0, offset=0.0):
        self.sounds.append(dict(path=str(path), at=at, gain_db=gain_db, offset=offset))

    # ---------- rendering ----------
    def _prepare(self):
        if self._prepared:
            return
        self.layers.sort(key=lambda layer: layer.z)
        for layer in self.layers:
            layer.prepare(self)
        self._zooms = []
        for z in self.zooms:
            t0 = self.out_time(z["at"])
            t1 = self.out_time(z["until"], "end") if z["until"] is not None else math.inf
            self._zooms.append(dict(z, t0=t0, t1=t1))
        W, H = self.size
        s = (max if self.fit == "cover" else min)(W / self.src_w, H / self.src_h)
        self._sharp = (0.35 if s > 1.6 else 0.0) if self.sharpen == "auto" else float(self.sharpen)
        self._prepared = True

    def _zoom_at(self, t):
        z, best, center = 1.0, 0.0, (0.5, 0.5)
        for zm in self._zooms:
            if t < zm["t0"]:
                continue
            e = ease_in_out((t - zm["t0"]) / zm["attack"]) if zm["attack"] else 1.0
            if t > zm["t1"]:
                e *= 1 - ease_in_out((t - zm["t1"]) / zm["release"]) if zm["release"] else 0.0
            k = (zm["scale"] - 1) * e
            z *= 1 + k
            if k > best:
                best, center = k, zm["center"]
        W, H = self.size
        return z, (center[0] * W, center[1] * H)

    def _matrix(self, t, fit):
        W, H = self.size
        w, h = self.src_w, self.src_h
        s0 = max(W / w, H / h) if fit == "cover" else min(W / w, H / h)
        hw, hh = W / s0 / 2, H / s0 / 2
        cx = min(max(self.focus[0] * w, hw), w - hw) if hw < w / 2 else w / 2
        cy = min(max(self.focus[1] * h, hh), h - hh) if hh < h / 2 else h / 2
        z, (px, py) = self._zoom_at(t)
        a = s0 * z
        return np.array([[a, 0, -a * cx + z * (W / 2 - px) + px], [0, a, -a * cy + z * (H / 2 - py) + py]])

    def _place(self, src, t):
        W, H = self.size
        if self.fit == "contain":
            bg = warp(src, self._matrix(t, "cover") / 8, (W // 8, H // 8))
            bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 4), (W, H), interpolation=cv2.INTER_LINEAR)
            out = warp(src, self._matrix(t, "contain"), (W, H), dst=cv2.convertScaleAbs(bg, alpha=0.6))
        else:
            out = warp(src, self._matrix(t, "cover"), (W, H))
        if self._sharp:
            out = cv2.addWeighted(out, 1 + self._sharp, cv2.GaussianBlur(out, (0, 0), 1.1), -self._sharp, 0)
        return out

    def _ensure_process(self):
        self._prepare()
        if self._pid != os.getpid():  # fresh decoders in every worker process
            self._pid = os.getpid()
            m = self.work / "master.mp4"
            self._readers = [FrameReader(m, self.src_w, self.src_h, self.fps, self.frames) for _ in range(2)]
            self._masks = mask.load(self.work)

    def frame_at(self, t):
        """THE render function: output time (seconds) -> BGR uint8 frame."""
        self._ensure_process()
        canvas, srcs = None, []
        for seg, ts, w in self.timemap.sources(t):
            idx = min(int(round(ts * float(self.fps))), self.frames - 1)
            img = self._place(self._readers[seg % 2].get(idx), t)
            canvas = img if canvas is None else cv2.addWeighted(canvas, 1 - smoothstep(w), img, smoothstep(w), 0)
            srcs.append((idx, w))
        ctx = Ctx(self, t, srcs)
        for layer in self.layers:
            layer.render(canvas, ctx)
        return canvas

    def _person_alpha(self, t, srcs):
        if self._masks is None:
            raise RuntimeError("no person masks yet - run: python -m vedit mask <project>")
        W, H = self.size
        mh, mw = self._masks.shape[1:]
        M = self._matrix(t, self.fit)
        M[:, 0] *= self.src_w / mw
        M[:, 1] *= self.src_h / mh
        acc = None
        for idx, w in srcs:
            m = cv2.warpAffine(np.asarray(self._masks[idx], np.float32) / 255.0, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            acc = m if acc is None else acc * (1 - smoothstep(w)) + m * smoothstep(w)
        return acc

    def render_audio(self, dst):
        sr = 48000
        src = self.work / "audio.wav"
        if src.exists():
            x, sr = audio.load_wav(src)
            out = audio.render(self.timemap, x, sr)
        else:
            out = np.zeros((int(round(self.timemap.duration * sr)), 2), np.float32)
        for s in self.sounds:
            audio.mix(out, sr, audio.decode(s["path"], sr, out.shape[1]), self.out_time(s["at"]) + s["offset"], s["gain_db"])
        raw = Path(f"{dst}.raw.wav")
        audio.save_wav(raw, out, sr)
        if self.loudness is None:
            os.replace(raw, dst)
        else:
            self.report["loudness_in"] = audio.master(raw, dst, self.loudness)
            raw.unlink()

    def render(self, name="v1", preview=False, workers=None, crf=17, preset="medium", log=print):
        self._prepare()
        self.out.mkdir(parents=True, exist_ok=True)
        fps = float(self.fps)
        n = max(1, int(round(self.timemap.duration * fps)))
        workers = workers or os.cpu_count() or 2
        parts = self.out / f".{name}.parts"
        shutil.rmtree(parts, ignore_errors=True)
        parts.mkdir()
        enc = ["-c:v", "libx264", "-preset", "ultrafast" if preview else preset, "-crf", str(26 if preview else crf),
               "-profile:v", "high", "-vf", f"scale=flags={SWS}:out_color_matrix=bt709:out_range=tv,format=yuv420p", *BT709,
               "-g", str(int(round(2 * fps))), "-threads", "2"]
        bounds = np.linspace(0, n, min(n, workers * 2) + 1).round().astype(int)
        tasks = [(int(a), int(b), str(parts / f"{k:03d}.mp4")) for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])) if b > a]
        global _JOB
        _JOB, t0, done = (self, enc), time.time(), 0
        with mp.get_context("fork").Pool(workers) as pool:
            for k in pool.imap_unordered(_render_part, tasks):
                done += k
                log(f"render {done}/{n} frames  ({done / (time.time() - t0):.1f} fps)")
        _JOB = None
        wav = self.out / f".{name}.wav"
        self.render_audio(wav)
        lst = parts / "list.txt"
        lst.write_text("".join(f"file '{t[2]}'\n" for t in tasks))
        dst = self.out / f"{name}.mp4"
        run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-map", "0:v:0",
             "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart",
             "-t", f"{n / fps:.6f}", dst])
        shutil.rmtree(parts)
        wav.unlink()
        tm = self.timemap
        (self.out / f"{name}.json").write_text(json.dumps(dict(
            format=self.format_name, size=self.size, fps=str(self.fps), frames=n, duration=round(tm.duration, 3),
            segments=[dict(src_in=round(s.src_in, 3), src_out=round(s.src_out, 3), hold=s.hold, xfade=s.xfade,
                           out_start=round(st, 3)) for s, st in zip(tm.segs, tm.starts)],
            report=self.report, seconds=round(time.time() - t0, 1)), indent=1, ensure_ascii=False))
        return dst

    def contact(self, video, per_sec=6):
        return contact.sheet(video, Path(video).with_suffix("").as_posix() + "_contact", per_sec=per_sec,
                             timemap=self.timemap)


_JOB = None


def _render_part(task):
    a, b, path = task
    p, enc = _JOB
    W, H = p.size
    proc = subprocess.Popen([FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", str(p.fps), "-i", "-", *enc, path], stdin=subprocess.PIPE)
    try:
        for i in range(a, b):
            proc.stdin.write(p.frame_at(i / float(p.fps)).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc:
        raise RuntimeError(f"encoder failed on frames {a}-{b}")
    return b - a


def load(root, **kw):
    """Project + its edit.py (a `build(p)` function describing the edit; extra kw, e.g. variant, go to build)."""
    p = Project(root)
    edit = Path(root) / "edit.py"
    if edit.exists():
        ns = {"__file__": str(edit)}
        exec(compile(edit.read_text(), str(edit), "exec"), ns)
        ns["build"](p, **kw)
    return p
