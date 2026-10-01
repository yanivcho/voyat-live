"""Download the models the engine needs (idempotent, resumable)."""
import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

HOME = Path(os.environ.get("VEDIT_HOME", Path.home() / ".cache" / "vedit"))
IVRIT_REPO = "ivrit-ai/whisper-large-v3-turbo-ggml"
SILERO_VERSION = "6.2.3"  # its ONNX interface is the one audio.vad_probs() speaks


def _get(url, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{dst}.part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if have and r.status == 206 else "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dst)
    return dst


def ivrit(dst=None):
    dst = Path(dst or HOME / "models" / "ivrit-large-v3-turbo.bin")
    if not dst.exists():
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{IVRIT_REPO}", timeout=30) as r:
            files = [s["rfilename"] for s in json.load(r).get("siblings", []) if s["rfilename"].endswith(".bin")]
        name = "ggml-model.bin" if "ggml-model.bin" in files else sorted(files)[0]
        _get(f"https://huggingface.co/{IVRIT_REPO}/resolve/main/{name}", dst)
    return dst


def silero(dst=None):
    dst = Path(dst or HOME / "models" / "silero_vad.onnx")
    if not dst.exists():
        with urllib.request.urlopen(f"https://pypi.org/pypi/silero-vad/{SILERO_VERSION}/json", timeout=30) as r:
            url = next(u["url"] for u in json.load(r)["urls"] if u["filename"].endswith(".whl"))
        with urllib.request.urlopen(url, timeout=120) as r:
            whl = zipfile.ZipFile(io.BytesIO(r.read()))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(whl.read("silero_vad/data/silero_vad.onnx"))
    return dst


def person_segmentation():
    from rembg import new_session
    new_session("u2net_human_seg")  # rembg downloads and caches the model on first use


if __name__ == "__main__":
    for name, fn in (("silero VAD", silero), ("u2net_human_seg", person_segmentation), ("ivrit.ai whisper", ivrit)):
        try:
            print(f"{name}: {fn() or 'ok'}")
        except Exception as e:  # keep going: each model is optional for some commands
            print(f"{name}: FAILED - {e}")
