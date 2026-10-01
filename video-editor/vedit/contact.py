"""Contact sheets for checking a render before showing it: N frames per second (one row = one second),
output time on every thumbnail (+ source time when a time map is given), and the soundtrack's waveform under
each row with red marks at the cuts. Labels are ASCII only, so cv2.putText is fine here."""
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .media import FFMPEG, probe


def sheet(video, out_dir, per_sec=6, secs_per_page=6, thumb_w=None, timemap=None):
    info = probe(video)
    W, H = info["width"], info["height"]
    tw = thumb_w or (176 if H > W else 300)
    th = int(round(tw * H / W / 2)) * 2
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-vf",
                          f"fps={per_sec}:start_time=0,scale={tw}:{th}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-"], capture_output=True, check=True).stdout
    thumbs = np.frombuffer(raw, np.uint8).reshape(-1, th, tw, 3)
    sr = 8000
    pcm = subprocess.run([FFMPEG, "-v", "error", "-i", str(video), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True).stdout
    snd = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    cuts = timemap.cut_points() if timemap else []
    gap, left, wave_h, head = 4, 46, 34, 26
    row_h = th + wave_h + gap * 2
    page_w = left + per_sec * (tw + gap) + gap
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds = int(np.ceil(len(thumbs) / per_sec))
    paths = []
    for p0 in range(0, seconds, secs_per_page):
        rows = min(secs_per_page, seconds - p0)
        page = np.full((head + rows * row_h, page_w, 3), 24, np.uint8)
        cv2.putText(page, f"{Path(video).name}  {p0}s-{p0 + rows}s  ({per_sec} fps, out time / src time)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        for r in range(rows):
            s = p0 + r
            y = head + r * row_h
            cv2.putText(page, f"{s}s", (6, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            for c in range(per_sec):
                k = s * per_sec + c
                if k >= len(thumbs):
                    break
                t = k / per_sec
                x = left + gap + c * (tw + gap)
                page[y:y + th, x:x + tw] = thumbs[k]
                lab = f"{t:.2f}"
                if timemap:
                    lab += f" s{timemap.sources(t)[-1][1]:.2f}"
                cv2.rectangle(page, (x, y), (x + 8 + 7 * len(lab), y + 15), (0, 0, 0), -1)
                cv2.putText(page, lab, (x + 3, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            # waveform of this second, aligned with the thumbnails
            wy, wx0, ww = y + th + gap, left + gap, per_sec * (tw + gap) - gap
            seg = snd[int(s * sr): int((s + 1) * sr)]
            mid = wy + wave_h // 2
            cv2.line(page, (wx0, mid), (wx0 + ww, mid), (70, 70, 70), 1)
            if len(seg):
                cols = np.array_split(seg, ww)
                for i, col in enumerate(cols):
                    if len(col):
                        a = int(min(1.0, np.abs(col).max()) * (wave_h // 2 - 1))
                        cv2.line(page, (wx0 + i, mid - a), (wx0 + i, mid + a), (110, 200, 120), 1)
            for ct in cuts:
                if s <= ct < s + 1:
                    cx = wx0 + int((ct - s) * ww)
                    cv2.line(page, (cx, wy), (cx, wy + wave_h), (0, 0, 255), 2)
        path = out_dir / f"contact_{p0:04d}s.png"
        cv2.imwrite(str(path), page)
        paths.append(path)
    return paths
