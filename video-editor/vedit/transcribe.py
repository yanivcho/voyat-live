"""Word-level transcription: whisper.cpp + the ivrit.ai Hebrew model (ivrit-large-v3-turbo, ggml)."""
import csv
import json
import os
from pathlib import Path

import numpy as np

from . import audio
from .media import run

HOME = Path(os.environ.get("VEDIT_HOME", Path.home() / ".cache" / "vedit"))
WHISPER = Path(os.environ.get("VEDIT_WHISPER_CLI", HOME / "bin" / "whisper-cli"))
MODEL = Path(os.environ.get("VEDIT_WHISPER_MODEL", HOME / "models" / "ivrit-large-v3-turbo.bin"))


def transcribe(work, model=None, lang="he", prompt=None, dtw=True, threads=None):
    work, model = Path(work), Path(model or MODEL)
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model} (setup.sh downloads it; needs huggingface.co access)")
    base = work / "whisper"
    cmd = [WHISPER, "-m", model, "-f", work / "audio16k.wav", "-l", lang, "-t", threads or os.cpu_count(),
           "-ojf", "-of", base, "-ml", "1", "-sow", "-np"]
    if dtw:  # DTW token timestamps (needs flash attention off)
        cmd += ["-dtw", "large.v3.turbo", "-nfa"]
    if prompt:  # names / jargon that should be spelled a certain way
        cmd += ["--prompt", prompt]
    run(cmd)
    words = parse(base.with_suffix(".json"))
    x16, _ = audio.load_wav(work / "audio16k.wav")
    flag_non_speech(words, x16[:, 0])
    save(work, words)
    return words


def parse(path):
    d = json.loads(Path(path).read_bytes().decode("utf-8", "replace"))
    words = []
    for seg in d.get("transcription", []):
        text = seg["text"].strip()
        if not text:
            continue
        if not any(ch.isalnum() for ch in text):  # punctuation-only piece belongs to the previous word
            if words:
                words[-1]["text"] += text
            continue
        toks = [t for t in seg.get("tokens", []) if not t["text"].startswith("[_")]
        dtw = [t["t_dtw"] / 100 for t in toks if t.get("t_dtw", -1) >= 0]
        start, end = seg["offsets"]["from"] / 1000, seg["offsets"]["to"] / 1000
        if dtw and abs(dtw[0] - start) < 0.6:  # DTW start is more precise when it agrees with the decoder
            start = dtw[0]
        p = float(np.mean([t["p"] for t in toks])) if toks else None
        words.append(dict(text=text, start=round(start, 3), end=round(end, 3), p=None if p is None else round(p, 3)))
    for a, b in zip(words, words[1:]):
        a["end"] = min(a["end"], b["start"])
    for w in words:
        w["end"] = round(max(w["end"], w["start"] + 0.05), 3)
    return words


def flag_non_speech(words, x16):
    """Mark words that fall where Silero VAD hears no speech (typical Whisper hallucinations)."""
    probs, hop = audio.vad_probs(x16), 512 / 16000
    for w in words:
        a = int(w["start"] / hop)
        seg = probs[a:max(a + 1, int(np.ceil(w["end"] / hop)))]
        if len(seg) and seg.max() < 0.3:
            w["flag"] = "no-speech"


def table(words):
    rows = ["| # | מילה | התחלה (שנ׳) | סוף (שנ׳) |", "|---:|---|---:|---:|"]
    for i, w in enumerate(words):
        rows.append(f"| {i} | {w['text']}{' ⚠' if w.get('flag') else ''} | {w['start']:.2f} | {w['end']:.2f} |")
    return "\n".join(rows)


def save(work, words):
    work = Path(work)
    (work / "words.json").write_text(json.dumps(dict(words=words), indent=1, ensure_ascii=False))
    (work / "words.md").write_text(table(words) + "\n")
    with open(work / "words.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["#", "word", "start", "end", "p", "flag"])
        for i, w in enumerate(words):
            wr.writerow([i, w["text"], w["start"], w["end"], w.get("p"), w.get("flag", "")])
