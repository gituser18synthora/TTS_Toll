#!/usr/bin/env bash
# One-time setup: Python venv + pinned deps, ffmpeg check, frontend build, DB init, model download.
#
#   ./setup.sh                 auto-detect CPU / NVIDIA GPU
#   ./setup.sh --cpu           force CPU torch wheels
#   ./setup.sh --gpu           force CUDA torch wheels (cu128; TORCH_VARIANT=cu126 for older drivers)
#   ./setup.sh --all-models    download whisper weights for both CPU and GPU sizes
#   ./setup.sh --no-models --no-frontend
#
# Env overrides: PYTHON (interpreter, default python3.11), VENV (default .venv), TORCH_VARIANT (cpu|cu126|cu128|cu130)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

VENV="${VENV:-.venv}"
PYTHON="${PYTHON:-}"
TORCH_VARIANT="${TORCH_VARIANT:-auto}"
DO_MODELS=1
DO_FRONTEND=1
ALL_MODELS=""

for arg in "$@"; do
  case "$arg" in
    --cpu) TORCH_VARIANT=cpu ;;
    --gpu) [ "$TORCH_VARIANT" = auto ] && TORCH_VARIANT=cu128 ;;
    --no-models) DO_MODELS=0 ;;
    --no-frontend) DO_FRONTEND=0 ;;
    --all-models) ALL_MODELS="--all" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- ffmpeg -------------------------------------------------------------------
say "Checking ffmpeg"
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  die "ffmpeg/ffprobe not found. Install it first:
    Ubuntu/Debian:  sudo apt-get install -y ffmpeg
    macOS:          brew install ffmpeg
    conda:          conda install -c conda-forge ffmpeg"
fi
ffmpeg -hide_banner -version | head -1

# --- python -------------------------------------------------------------------
say "Python environment ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
  if [ -z "$PYTHON" ]; then
    for cand in python3.11 python3.12; do
      if command -v "$cand" >/dev/null; then PYTHON="$cand"; break; fi
    done
  fi
  if [ -n "$PYTHON" ]; then
    "$PYTHON" -m venv "$VENV" || die "could not create venv with $PYTHON (on Debian/Ubuntu: sudo apt-get install ${PYTHON}-venv)"
  elif command -v uv >/dev/null; then
    echo "python3.11 not found; using uv to provision it"
    uv venv --python 3.11 --seed "$VENV"
  else
    die "Python 3.11 not found. Install it (e.g. 'sudo apt-get install python3.11 python3.11-venv',
    deadsnakes PPA, conda, or 'curl -LsSf https://astral.sh/uv/install.sh | sh') or set PYTHON=/path/to/python3.11"
  fi
fi
PY="$ROOT/$VENV/bin/python"
"$PY" --version
"$PY" -m pip install --quiet --upgrade pip wheel

# --- torch (CPU or CUDA wheels) -------------------------------------------------
if [ "$TORCH_VARIANT" = auto ]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    CUDA_DRV=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || echo 0)
    # ctranslate2 (faster-whisper) is built against CUDA 12, so stay on cu12x wheels.
    if awk "BEGIN{exit !($CUDA_DRV >= 12.8)}"; then TORCH_VARIANT=cu128; else TORCH_VARIANT=cu126; fi
    echo "NVIDIA driver supports CUDA $CUDA_DRV -> torch variant $TORCH_VARIANT"
  else
    TORCH_VARIANT=cpu
    echo "no NVIDIA GPU detected -> CPU torch"
  fi
fi
say "Installing torch ($TORCH_VARIANT)"
"$PY" -m pip install --quiet -r requirements-torch.txt --index-url "https://download.pytorch.org/whl/$TORCH_VARIANT"

say "Installing Python requirements (pinned)"
"$PY" -m pip install --quiet -r requirements.txt -c constraints.txt
"$PY" -m pip install --quiet --no-deps -r requirements-nodeps.txt
# DeepFilterNet's outdated numpy/packaging pins are expected to show up here
"$PY" -m pip check | grep -v "^deepfilternet" || true

# --- frontend -----------------------------------------------------------------------
if [ "$DO_FRONTEND" = 1 ]; then
  if command -v npm >/dev/null; then
    say "Frontend (npm ci + build)"
    (cd frontend && { [ -f package-lock.json ] && npm ci --no-fund --no-audit || npm install --no-fund --no-audit; } && npm run build)
  else
    echo "WARNING: npm not found; skipping frontend. Install Node.js >= 20.19 (e.g. via nvm) and re-run, or use Docker."
  fi
fi

# --- data folder, database, models ---------------------------------------------------
say "Initialising DATA_ROOT and database"
(cd backend && "$PY" -m app.cli init)

if [ "$DO_MODELS" = 1 ]; then
  say "Downloading model weights into ./models"
  (cd backend && "$PY" -m app.cli download-models $ALL_MODELS)
fi

say "Environment check"
(cd backend && "$PY" -m app.cli doctor)

cat <<EOF

Setup complete.
  Start everything (API + worker + UI):   ./run_dev.sh
  Then open:                              http://localhost:5173
EOF
