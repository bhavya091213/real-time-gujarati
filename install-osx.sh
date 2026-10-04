#!/usr/bin/env bash
# guju-sub installer for macOS (Apple Silicon or Intel).
#
#   ./install-osx.sh              # venv + runtime deps + pre-download models
#   ./install-osx.sh --dev        # also install pytest/httpx/ruff and run the test suite
#   ./install-osx.sh --skip-models
#
# Safe to re-run: reuses an existing .venv and only installs what is missing.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DEV=0
SKIP_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --dev) DEV=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

MIN_MINOR=11   # requires-python >= 3.11 (pyproject.toml)

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. find a suitable Python -------------------------------------------
find_python() {
  local cand
  for cand in python3.13 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      local minor
      minor="$("$cand" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
      if [ "$minor" -ge "$MIN_MINOR" ]; then
        command -v "$cand"
        return 0
      fi
    fi
  done
  return 1
}

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
  say "reusing existing .venv ($("$PY" --version))"
else
  if ! SYS_PY="$(find_python)"; then
    if command -v brew >/dev/null 2>&1; then
      say "no Python >= 3.$MIN_MINOR found; installing python@3.12 with Homebrew"
      brew install python@3.12
      SYS_PY="$(find_python)" || fail "python@3.12 installed but not on PATH; open a new shell and re-run"
    else
      fail "Python >= 3.$MIN_MINOR not found. Install Homebrew (https://brew.sh) or python.org, then re-run."
    fi
  fi
  say "creating .venv with $SYS_PY ($("$SYS_PY" --version))"
  "$SYS_PY" -m venv .venv
  PY=.venv/bin/python
fi

# ---- 2. install dependencies ---------------------------------------------
say "upgrading pip"
"$PY" -m pip install --quiet --upgrade pip

if [ "$DEV" -eq 1 ]; then
  say "installing runtime + dev dependencies (requirements-dev.txt)"
  "$PY" -m pip install -r requirements-dev.txt
else
  say "installing runtime dependencies (requirements.txt)"
  "$PY" -m pip install -r requirements.txt
fi

say "verifying imports"
"$PY" -c 'from gujusub.translator import Translator; from gujusub.asr_engine import ASREngine' \
  || fail "import check failed; see the traceback above"

# ---- 3. pre-download models (~1 GB; hub cache + real-file copy in ~/.cache/gujusub)
if [ "$SKIP_MODELS" -eq 0 ]; then
  say "downloading models (skip with --skip-models)"
  "$PY" tools/prefetch_models.py || fail "model download failed; check your network and re-run"

  say "loading models once to verify they work (ASR + translator, ~30 s)"
  "$PY" tools/prefetch_models.py --verify || fail "model load check failed; see the traceback above"
fi

# ---- 4. optional: run the test suite ---------------------------------------
if [ "$DEV" -eq 1 ]; then
  say "running tests"
  "$PY" -m pytest
fi

cat <<EOF

Done.

Next: ./start.sh --open      (starts the server and opens the mic + display pages)

Add --device mps on Apple Silicon to run the ASR on the GPU (./start.sh --open --device mps).
EOF
