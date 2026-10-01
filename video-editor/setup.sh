#!/usr/bin/env bash
# Installs everything the engine needs. Safe to re-run (skips what is already there).
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VEDIT_VENV:-/opt/vedit-venv}"
HOME_DIR="${VEDIT_HOME:-$HOME/.cache/vedit}"
mkdir -p "$HOME_DIR/bin" "$HOME_DIR/models" "$HOME_DIR/src"

SUDO=$(command -v sudo || true)
command -v ffmpeg >/dev/null || { $SUDO apt-get update && $SUDO apt-get install -y ffmpeg; }
ldconfig -p | grep -q libfribidi || $SUDO apt-get install -y libfribidi0   # Hebrew bidi for Pillow/RAQM
command -v espeak-ng >/dev/null || $SUDO apt-get install -y espeak-ng || true  # only for selftest.py

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r requirements.txt

if [ ! -x "$HOME_DIR/bin/whisper-cli" ]; then
  [ -d "$HOME_DIR/src/whisper.cpp" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$HOME_DIR/src/whisper.cpp"
  cmake -S "$HOME_DIR/src/whisper.cpp" -B "$HOME_DIR/src/whisper.cpp/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF
  cmake --build "$HOME_DIR/src/whisper.cpp/build" -j"$(nproc)" --target whisper-cli
  cp "$HOME_DIR/src/whisper.cpp/build/bin/whisper-cli" "$HOME_DIR/bin/"
fi

"$VENV/bin/python" -c "from PIL import features; assert features.check('raqm'), 'Pillow has no RAQM - Hebrew would be reversed'"
"$VENV/bin/python" -m vedit.models
echo "ready: $VENV/bin/python -m vedit --help"
