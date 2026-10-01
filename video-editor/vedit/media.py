"""Media I/O: probe a source, ingest it into a clean working master, and read frames.

Ingest normalizes every phone/camera quirk once, so the rest of the engine can rely on:
  * constant frame rate: frame i is exactly at i / fps seconds
  * rotation already applied, HDR (HLG/PQ) tone-mapped to SDR BT.709
  * audio.wav (48 kHz) and audio16k.wav (mono, for transcription) start at the same t=0 as the video
"""
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
STD_FPS = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
           Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60)]
SWS = "accurate_rnd+full_chroma_int"  # without these flags every RGB<->YUV trip darkens colors by 2-4 levels
BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}\n{r.stderr[-4000:]}")
    return r.stdout


def probe(path):
    d = json.loads(run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]))
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise ValueError(f"no video stream in {path}")
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if not rot and "rotate" in (v.get("tags") or {}):
        rot = int(v["tags"]["rotate"])
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = Fraction(v.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(v.get("r_frame_rate") or "30/1")
    return dict(width=w, height=h, rotation=rot, fps=str(fps), duration=float(d["format"].get("duration") or 0),
                codec=v.get("codec_name"), pix_fmt=v.get("pix_fmt"), color_transfer=v.get("color_transfer"),
                color_primaries=v.get("color_primaries"), has_audio=a is not None,
                audio_rate=int(a["sample_rate"]) if a else None, audio_channels=int(a["channels"]) if a else None)


def working_fps(src_fps, override=None):
    """Closest standard rate; 50/60 fps sources are edited at 25/30 unless overridden."""
    if override:
        return Fraction(override).limit_denominator(1001)
    f = min(STD_FPS, key=lambda s: abs(float(s) - float(Fraction(src_fps))))
    return f / 2 if float(f) > 31 else f


def ingest(src, work, fps=None, max_side=3840):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    wf = working_fps(info["fps"], fps)
    s = min(1.0, max_side / max(info["width"], info["height"]))
    W, H = int(round(info["width"] * s / 2)) * 2, int(round(info["height"] * s / 2)) * 2
    vf = [f"fps=fps={wf}:start_time=0"]
    if info["color_transfer"] in ("arib-std-b67", "smpte2084"):  # HLG / PQ (e.g. iPhone HDR) -> SDR
        vf += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
               "tonemap=tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv"]
    vf += [f"scale={W}:{H}:flags=lanczos+{SWS}:out_color_matrix=bt709:out_range=tv", "format=yuv420p"]
    run([FFMPEG, "-y", "-v", "error", "-i", src, "-map", "0:v:0", "-vf", ",".join(vf), "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "12", "-g", "15", "-bf", "0", *BT709, "-an", work / "master.mp4"])
    if info["has_audio"]:
        ch = min(info["audio_channels"] or 1, 2)
        run([FFMPEG, "-y", "-v", "error", "-i", src, "-map", "0:a:0", "-af", "aresample=async=1:first_pts=0",
             "-ar", "48000", "-ac", ch, "-c:a", "pcm_s16le", work / "audio.wav"])
        run([FFMPEG, "-y", "-v", "error", "-i", work / "audio.wav", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", work / "audio16k.wav"])
    n = int(run([FFPROBE, "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                 "stream=nb_read_packets", "-of", "csv=p=0", work / "master.mp4"]).strip().split(",")[0])
    meta = dict(source=str(Path(src).resolve()), width=W, height=H, fps=str(wf), frames=n,
                duration=n / float(wf), has_audio=info["has_audio"], source_probe=info)
    (work / "info.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


class FrameReader:
    """Random access to master.mp4 frames (BGR uint8), cheap when reading forward."""

    def __init__(self, path, width, height, fps, frames, size=None):
        self.path, self.fps, self.n = str(path), float(Fraction(fps)), frames
        self.w, self.h = size or (width, height)
        self.scaled = size is not None
        self.nbytes = self.w * self.h * 3
        self.proc, self.pos, self.idx, self.frame = None, 0, -1, None

    def _open(self, idx):
        self.close()
        vf = f"scale={self.w}:{self.h}:flags=area+{SWS}:" if self.scaled else f"scale=flags={SWS}:"
        vf += "in_color_matrix=bt709:in_range=tv,format=bgr24"
        cmd = [FFMPEG, "-v", "error", "-ss", f"{max(0.0, (idx - 0.5) / self.fps):.6f}", "-i", self.path,
               "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.nbytes)
        self.pos = idx

    def get(self, idx):
        idx = min(max(int(idx), 0), self.n - 1)
        if idx == self.idx:
            return self.frame
        if self.proc is None or idx < self.pos or idx > self.pos + 3 * self.fps:
            self._open(idx)
        buf = None
        while self.pos <= idx:
            buf = self.proc.stdout.read(self.nbytes)
            if len(buf) < self.nbytes:
                buf = None
                break
            self.pos += 1
        if buf is None:  # ran past the real end of the stream
            if self.frame is None:
                raise IOError(f"cannot read frame {idx} of {self.path}")
            return self.frame
        self.idx, self.frame = idx, np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        return self.frame

    def close(self):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
            self.proc = None
