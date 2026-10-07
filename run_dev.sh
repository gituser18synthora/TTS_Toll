#!/usr/bin/env bash
# Start the API (auto-reload), the job worker and the Vite dev server together.
# Ctrl+C stops all three; an in-flight job is re-queued and resumes next time.
#
#   ./run_dev.sh            dev mode: UI on http://localhost:5173 (hot reload), API on :8000
#   ./run_dev.sh --prod     no Vite: builds the UI once and serves everything from the API port
#
# Env: VENV (default .venv), VITE_HOST / VITE_PORT, PORTAL_CONFIG, PORTAL_DATA_ROOT
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="$ROOT/${VENV:-.venv}/bin/python"
MODE=dev
[ "${1:-}" = "--prod" ] && MODE=prod

[ -x "$PY" ] || { echo "No virtualenv at ${VENV:-.venv}. Run ./setup.sh first."; exit 1; }
export PYTHONUNBUFFERED=1

# Make pip-installed CUDA libraries (cuBLAS/cuDNN from the torch wheels) visible to ctranslate2.
NV_LIBS=$("$PY" - <<'PYEOF'
import importlib.util
dirs = []
for mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib", "nvidia.cuda_runtime.lib", "nvidia.cuda_nvrtc.lib"):
    try:
        spec = importlib.util.find_spec(mod)
    except ModuleNotFoundError:
        spec = None
    if spec and spec.submodule_search_locations:
        dirs.extend(spec.submodule_search_locations)
print(":".join(dirs))
PYEOF
)
[ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="$NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

read -r HOST PORT < <(cd backend && "$PY" -c "from app.config import get_settings as g; s=g(); print(s.server.host, s.server.port)")

# On exit stop only what this script started (its own process tree). `kill 0` is not used:
# outside an interactive terminal the process group can include whoever launched us.
kill_tree() {
  local pid=$1 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
  # the `sed` log prefixers exit by themselves once their input closes, keeping final log lines
  [ "$(ps -o comm= -p "$pid" 2>/dev/null)" = sed ] || kill -TERM "$pid" 2>/dev/null || true
}
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "stopping… (an in-flight job is re-queued and resumes on the next start)"
  for pid in $(pgrep -P $$ 2>/dev/null); do kill_tree "$pid"; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

prefix() { sed -u "s/^/$1 /"; }

if [ "$MODE" = prod ]; then
  (cd frontend && { [ -d node_modules ] || npm ci --no-fund --no-audit; } && npm run build) 2>&1 | prefix "[web]"
  (cd backend && exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT") 2>&1 | prefix "[api]" &
else
  (cd backend && exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload --reload-dir app) 2>&1 | prefix "[api]" &
fi

# wait for the API (it creates/migrates the DB) before starting the worker
for _ in $(seq 1 60); do
  curl -sf "http://$HOST:$PORT/api/health" >/dev/null 2>&1 && break
  sleep 0.5
done
(cd backend && exec "$PY" -m app.worker) 2>&1 | prefix "[worker]" &

if [ "$MODE" = dev ]; then
  (cd frontend && { [ -d node_modules ] || npm ci --no-fund --no-audit; } && exec npm run dev -- --clearScreen false) 2>&1 | prefix "[web]" &
  sleep 2
  echo
  echo "  UI:   http://localhost:${VITE_PORT:-5173}"
  echo "  API:  http://$HOST:$PORT/docs"
else
  echo
  echo "  UI + API:  http://$HOST:$PORT"
fi
echo "  logs: data/logs/   (Ctrl+C to stop)"
echo
wait
