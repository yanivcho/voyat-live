"""End-to-end self test with known truth:  python selftest.py [workdir]

Builds a synthetic phone clip: Hebrew speech (espeak-ng) with pauses of known length, every frame stamped with
its frame number as a barcode, flat color patches, stored sideways with rotation metadata. Then runs the real
pipeline (ingest -> VAD silence cut with soft transitions -> animated captions -> parallel render -> contact
sheet) and checks the result against the truth:
  picture  every output frame shows exactly the source frame the time map says (no drift, drops or dups)
  sound    every phrase starts in the output exactly where the time map puts it
  color    patches keep their values through decode -> composite -> encode
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vedit import audio, load, media  # noqa: E402

FPS, W, H, BITS, SR = 30, 720, 1280, 12, 48000
PHRASES = ["שלום לכולם וברוכים הבאים", "היום נדבר", "על עריכת וידאו", "בעברית מימין לשמאל", "עם Voyat 2026"]
GAPS = [0.8, 1.2, 0.25, 2.5, 0.6, 1.2]  # lead, the four pauses between phrases, tail
PATCHES = [(40, 40, 200), (40, 180, 60), (200, 120, 30), (128, 128, 128), (235, 235, 235)]  # BGR


def speech(text, tmp):
    wav = tmp / "tts.wav"
    subprocess.run(["espeak-ng", "-v", "he", "-s", "150", "-w", str(wav), text], check=True)
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(wav), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    on = np.flatnonzero(np.abs(x) > 10 ** (-40 / 20))
    return x[on[0]:on[-1] + 1] * 0.7


def make_source(tmp):
    rng = np.random.default_rng(7)
    parts, truth, t = [], [], 0.0
    for i, text in enumerate(PHRASES):
        parts.append(np.zeros(int(GAPS[i] * SR), np.float32))
        t += GAPS[i]
        x = speech(text, tmp)
        truth.append(dict(text=text, start=t, end=t + len(x) / SR))
        parts.append(x)
        t += len(x) / SR
    parts.append(np.zeros(int(GAPS[-1] * SR), np.float32))
    a = np.concatenate(parts)
    a += rng.normal(0, 10 ** (-58 / 20), len(a)).astype(np.float32)  # room noise floor
    audio.save_wav(tmp / "a.wav", a[:, None], SR)
    n = int(len(a) / SR * FPS)
    enc = subprocess.Popen([media.FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{H}x{W}",
                            "-r", str(FPS), "-i", "-", "-i", str(tmp / "a.wav"), "-c:v", "libx264", "-crf", "14",
                            "-pix_fmt", "yuv420p", "-vf", f"scale=flags={media.SWS}:out_color_matrix=bt709:out_range=tv", *media.BT709,
                            "-c:a", "aac", "-b:a", "192k", "-shortest", str(tmp / "flat.mp4")], stdin=subprocess.PIPE)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    for i in range(n):
        ph = i / n
        img = ((1 - yy) * np.array([90, 60 + 80 * ph, 40]) + yy * np.array([30, 40, 90 + 100 * ph])).astype(np.uint8)
        img = np.repeat(img, W, axis=1)
        bw = (W - 80) // BITS
        for k in range(BITS):
            img[60:120, 40 + k * bw:40 + (k + 1) * bw] = 255 if (i >> k) & 1 else 0
        cv2.putText(img, f"SRC {i / FPS:05.2f}  #{i:03d}", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.circle(img, (int(W / 2 + 250 * np.sin(i / 15)), 760), 50, (0, 220, 255), -1)
        for k, c in enumerate(PATCHES):
            img[1100:1200, 40 + k * 130:140 + k * 130] = c
        enc.stdin.write(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE).tobytes())  # stored sideways, like a phone
    enc.stdin.close()
    assert enc.wait() == 0
    media.run([media.FFMPEG, "-y", "-v", "error", "-display_rotation", "90", "-i", tmp / "flat.mp4", "-c", "copy",
               tmp / "source.mp4"])
    return tmp / "source.mp4", truth


def fake_words(truth):
    """Word timings spread over each phrase by word length (the transcript stand-in)."""
    words = []
    for ph in truth:
        toks = ph["text"].split()
        L = sum(len(w) for w in toks)
        t = ph["start"]
        for w in toks:
            d = (ph["end"] - ph["start"]) * len(w) / L
            words.append(dict(text=w, start=round(t, 3), end=round(t + d, 3)))
            t += d
    return words


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="vedit-selftest-"))
    tmp, proj = base / "tmp", base / "selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    src, truth = make_source(tmp)
    probe = media.probe(src)
    assert (probe["width"], probe["height"]) == (W, H), f"rotation not detected: {probe}"
    info = media.ingest(src, proj / "work")
    assert (info["width"], info["height"], info["fps"]) == (W, H, "30"), info
    (proj / "work" / "words.json").write_text(json.dumps(dict(words=fake_words(truth)), ensure_ascii=False))
    (proj / "edit.py").write_text("def build(p):\n    p.set_format('vertical')\n    p.set_style('bold')\n"
                                  "    p.cut_silences()\n    p.captions()\n")
    p = load(proj)
    out = p.render("v1", log=lambda s: None)
    tm, fps = p.timemap, 30.0
    print("silence cut:", {k: v for k, v in p.report["silence_cut"].items() if k != "speech"})
    print("speech found by VAD+words:", p.report["silence_cut"]["speech"])

    # --- picture: decode every output frame's barcode
    vi = media.probe(out)
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 1920, 1080)
    s = 1080 / W
    bw = (W - 80) // BITS
    checked = bad = 0
    for k, fr in enumerate(frames):
        srcs = tm.sources(k / fps)
        if len(srcs) > 1:
            continue  # inside a dissolve two frames are mixed
        want = min(int(round(srcs[0][1] * fps)), info["frames"] - 1)
        got = sum(1 << b for b in range(BITS)
                  if fr[int(90 * s - 8):int(90 * s + 8), int((40 + (b + 0.5) * bw) * s) - 8:int((40 + (b + 0.5) * bw) * s) + 8].mean() > 128)
        checked += 1
        bad += got != want
    print(f"picture sync: {checked - bad}/{checked} frames exact; output {len(frames)} frames, expected {int(round(tm.duration * fps))}")

    # --- sound: phrase onsets in the output vs the time map
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-i", str(out), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    y = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    env = np.sqrt(np.convolve(y ** 2, np.ones(240) / 240, "same"))
    thr = env.max() * 10 ** (-30 / 20)
    errs = []
    for ph in truth:
        exp = tm.to_out(ph["start"])
        a = int((exp - 0.15) * SR)
        on = a + int(np.argmax(env[a:a + int(0.4 * SR)] > thr))
        errs.append((on / SR - exp) * 1000)
    print("phrase onset error (ms):", [round(e, 1) for e in errs])

    # --- color patches
    raw = subprocess.run([media.FFMPEG, "-v", "error", "-ss", "1.0", "-i", str(out), "-frames:v", "1", "-vf",
                          f"scale=flags={media.SWS}:in_color_matrix=bt709:in_range=tv", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(1920, 1080, 3)
    cerr = max(np.abs(fr[int(1150 * s) - 10:int(1150 * s) + 10, int((90 + k * 130) * s) - 10:int((90 + k * 130) * s) + 10]
                      .reshape(-1, 3).mean(0) - np.array(c)).max() for k, c in enumerate(PATCHES))
    print(f"color max error: {cerr:.1f} levels")
    sheets = p.contact(out)
    print("contact sheets:", *sheets)
    ok = bad == 0 and checked > 0 and max(abs(e) for e in errs) <= 25 and cerr <= 4 and vi["width"] == 1080
    print("SELFTEST", "PASSED" if ok else "FAILED", "-", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
