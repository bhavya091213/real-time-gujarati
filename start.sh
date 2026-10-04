#!/usr/bin/env bash
# guju-sub launcher for macOS. Installs on first run, then starts the server.
#
#   ./start.sh [script flags] [server flags]
#
# Script flags:
#   --setup             run install-osx.sh first even if .venv exists (idempotent)
#   --skip-models       passed to the installer when it runs
#   --open              open the mic page and /display in the browser once ready
#   --display-only      open only /display once ready
#   --threads N         export GUJUSUB_THREADS=N
#   --model-dir PATH    export GUJUSUB_MODEL_DIR=PATH
#   --replay FILE ...   run tools/replay.py FILE ... instead of the server
#   --verify            load both models once (tools/prefetch_models.py --verify) and exit
#   --keep-port         do not stop an existing listener on the server port first
#   --test              run pytest and exit
#   -h, --help          this text plus the live server flag list
#
# Before starting the server, anything already listening on the port (--port,
# default 8765) is stopped, unless --keep-port is given.
#
# Any other flag (--device, --lang, --port, --no-translate, --no-filter, ...)
# is passed to the server unchanged.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY=.venv/bin/python
SETUP=0
SKIP_MODELS=0
OPEN_MAIN=0
OPEN_DISPLAY=0
PORT=8765
READY_TIMEOUT_S=90
KEEP_PORT=0
SERVER_ARGS=()

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

need_value() { [ "$#" -ge 2 ] || fail "$1 needs a value"; }

usage() {
  sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  if [ -x "$PY" ]; then
    echo "Server flags (python -m gujusub.server --help):"
    "$PY" -m gujusub.server --help 2>/dev/null | sed 's/^/  /' || true
  else
    echo "(server flags are listed once .venv exists; run ./start.sh --setup)"
  fi
}

port_pids() {
  # PIDs listening on $PORT, minus this script and its parent (lsof exits 1 if none).
  local pid
  for pid in $(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true); do
    [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] || printf '%s ' "$pid"
  done
}

free_port() {
  local pids
  pids=$(port_pids)
  [ -n "$pids" ] || return 0
  say "stopping process(es) on port $PORT: $pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  for ((i = 0; i < 6; i++)); do
    [ -z "$(port_pids)" ] && return 0
    sleep 0.5
  done
  pids=$(port_pids)
  # shellcheck disable=SC2086
  [ -z "$pids" ] || kill -9 $pids 2>/dev/null || true
  sleep 0.5
  pids=$(port_pids)
  [ -z "$pids" ] || fail "port $PORT is still in use by PID(s) $pids; stop it or pass --keep-port"
}

ensure_venv() {
  if [ "$SETUP" -eq 1 ] || [ ! -x "$PY" ]; then
    local installer_args=()
    [ "$SKIP_MODELS" -eq 1 ] && installer_args+=(--skip-models)
    say "running installer"
    ./install-osx.sh ${installer_args[@]+"${installer_args[@]}"}
  fi
  [ -x "$PY" ] || fail ".venv/bin/python missing after install"
}

# ---- parse arguments --------------------------------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --setup) SETUP=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --open) OPEN_MAIN=1; OPEN_DISPLAY=1 ;;
    --display-only) OPEN_DISPLAY=1 ;;
    --keep-port) KEEP_PORT=1 ;;
    --threads) need_value "$@"; export GUJUSUB_THREADS="$2"; shift ;;
    --model-dir) need_value "$@"; export GUJUSUB_MODEL_DIR="$2"; shift ;;
    --verify)
      ensure_venv
      exec "$PY" tools/prefetch_models.py --verify ;;
    --test)
      ensure_venv
      exec "$PY" -m pytest ;;
    --replay)
      need_value "$@"; shift
      ensure_venv
      exec "$PY" tools/replay.py "$@" ;;
    --port)
      need_value "$@"; PORT="$2"; SERVER_ARGS+=("$1" "$2"); shift ;;
    --port=*)
      PORT="${1#--port=}"; SERVER_ARGS+=("$1") ;;
    *) SERVER_ARGS+=("$1") ;;
  esac
  shift
done

ensure_venv

[ "$KEEP_PORT" -eq 1 ] || free_port

# ---- run ----------------------------------------------------------------------
if [ "$OPEN_MAIN" -eq 0 ] && [ "$OPEN_DISPLAY" -eq 0 ]; then
  exec "$PY" -m gujusub.server ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"}
fi

"$PY" -m gujusub.server ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"} &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM

say "waiting for the server on port $PORT (models load, up to ${READY_TIMEOUT_S}s)"
ready=0
for ((i = 0; i < READY_TIMEOUT_S; i++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" || true
    fail "server exited before becoming ready"
  fi
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/" || true)" = "200" ]; then
    ready=1
    break
  fi
  sleep 1
done
[ "$ready" -eq 1 ] || fail "server not ready after ${READY_TIMEOUT_S}s"

say "ready"
[ "$OPEN_MAIN" -eq 1 ] && open "http://localhost:$PORT/"
[ "$OPEN_DISPLAY" -eq 1 ] && open "http://localhost:$PORT/display"

wait "$SERVER_PID"
