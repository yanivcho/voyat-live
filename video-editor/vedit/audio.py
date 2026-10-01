"""Audio: WAV I/O, speech detection (Silero VAD + energy), rendering through the time map, mastering."""
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from .media import FFMPEG, run

MODELS = Path.home() / ".cache" / "vedit" / "models"


def load_wav(path):
    with wave.open(str(path), "rb") as f:
        assert f.getsampwidth() == 2, "expected 16-bit PCM"
        sr, ch = f.getframerate(), f.getnchannels()
        x = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch), sr


def save_wav(path, x, sr):
    y = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(y.shape[1])
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(y.tobytes())


def decode(path, sr=48000, channels=2):
    """Any audio/video file -> float32 [n, channels] (for sound effects and music)."""
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "s16le", "-ac", str(channels),
                          "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32).reshape(-1, channels) / 32768.0


def _runs(mask):
    """[(start, end)] index runs where mask is True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def vad_probs(x16k):
    """Silero VAD speech probability for every 32 ms (512 samples at 16 kHz). Robust to wind/sea/crowd noise."""
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = o.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(MODELS / "silero_vad.onnx"), sess_options=o, providers=["CPUExecutionProvider"])
    state, ctx, out = np.zeros((2, 1, 128), np.float32), np.zeros((1, 64), np.float32), []
    sr = np.array(16000, np.int64)
    for i in range(0, len(x16k) - 511, 512):
        inp = np.concatenate([ctx, x16k[None, i:i + 512]], axis=1)
        p, state = sess.run(None, {"input": inp, "state": state, "sr": sr})
        ctx = inp[:, -64:]
        out.append(float(p[0, 0]))
    return np.array(out, np.float32)


def speech_regions(x16k, words=(), threshold=0.5, min_speech=0.15, merge_gap=0.12, pad=0.03):
    """Speech regions [(start, end)] in seconds: VAD regions, widened to cover every transcribed word."""
    hop = 512 / 16000
    probs = vad_probs(x16k)
    # hysteresis: enter above threshold, stay while above threshold - 0.15 (Silero's recommended scheme)
    on, active = np.zeros(len(probs), bool), False
    for i, p in enumerate(probs):
        active = p >= threshold or (active and p >= threshold - 0.15)
        on[i] = active
    regions = [[a * hop, b * hop] for a, b in _runs(on) if (b - a) * hop >= min_speech]
    for w in words:
        if not w.get("flag"):
            regions.append([w["start"], w["end"] + 0.02])
    regions.sort()
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(float(max(0.0, s - pad)), float(e + pad)) for s, e in merged], probs


def render(tm, x, sr, micro=0.004):
    """Build the output soundtrack by cutting the source audio with the same time map as the picture.
    Soft transitions get an equal-power crossfade; hard joints get a few-ms fade to avoid clicks."""
    n_out = int(round(tm.duration * sr))
    out = np.zeros((n_out, x.shape[1]), np.float32)
    segs = tm.segs

    def joined(a, b):
        return a.hold == 0 and b.hold == 0 and abs(a.src_out - b.src_in) < 1e-6

    for i, s in enumerate(segs):
        if s.hold:
            continue  # freeze frames are silent; add music/sfx on top if needed
        o, L = int(round(tm.starts[i] * sr)), int(round(s.dur * sr))
        L = min(L, n_out - o)
        if L <= 0:
            continue
        a = int(round(s.src_in * sr))
        chunk = np.zeros((L, x.shape[1]), np.float32)
        part = x[a:a + L]
        chunk[:len(part)] = part
        env = np.ones(L, np.float32)
        fi = int(round(s.xfade * sr)) if i else 0
        if i and not fi and not joined(segs[i - 1], s):
            fi = int(micro * sr)
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        fo = int(round(nxt.xfade * sr)) if nxt else 0
        if nxt and not fo and not joined(s, nxt):
            fo = int(micro * sr)
        if fi:
            env[:fi] *= np.sin(0.5 * np.pi * np.linspace(0, 1, fi, dtype=np.float32))
        if fo:
            env[L - fo:] *= np.cos(0.5 * np.pi * np.linspace(0, 1, fo, dtype=np.float32))
        out[o:o + L] += chunk * env[:, None]
    return out


def mix(out, sr, clip, at, gain_db=0.0, fade=0.01):
    """Add a sound (float32 [n, ch]) to the output track at output time `at`."""
    o = int(round(at * sr))
    if o >= len(out):
        return out
    c = clip[: len(out) - max(o, 0)] * (10 ** (gain_db / 20))
    if o < 0:
        c, o = c[-o:], 0
    f = min(int(fade * sr), len(c) // 2)
    if f:
        c[-f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
    ch = out.shape[1]
    out[o:o + len(c)] += c[:, :ch] if c.shape[1] >= ch else np.repeat(c[:, :1], ch, axis=1)
    return out


def master(src, dst, lufs=-14.0, true_peak=-1.0, highpass=70):
    """Two-pass EBU R128 loudness normalization (social platforms ~ -14 LUFS), with a rumble high-pass."""
    pre = f"highpass=f={highpass}," if highpass else ""
    base = f"loudnorm=I={lufs}:TP={true_peak}:LRA=11"
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(src), "-af", f"{pre}{base}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    try:
        m = json.loads(r.stderr[r.stderr.rindex("{"): r.stderr.rindex("}") + 1])
        ok = float(m["input_i"]) > -70
    except (ValueError, KeyError):
        ok = False
    af = pre + (f"{base}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
                f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true" if ok else "anull")
    run([FFMPEG, "-y", "-v", "error", "-i", src, "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", dst])
    return m if ok else None
