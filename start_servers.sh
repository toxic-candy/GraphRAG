#!/usr/bin/env bash
# start_servers.sh — Start the GraphRAG API + UI dev servers
# Usage: ./start_servers.sh [--stop]

ROOT="$(cd "$(dirname "$0")" && pwd)"
API_PID_FILE="/tmp/graphrag_api.pid"
UI_PID_FILE="/tmp/graphrag_ui.pid"
API_LOG="/tmp/graphrag_api.log"
UI_LOG="/tmp/graphrag_ui.log"

stop_servers() {
  echo "Stopping servers…"
  if [ -f "$API_PID_FILE" ]; then
    kill "$(cat $API_PID_FILE)" 2>/dev/null && echo "  ✓ API server stopped"
    rm -f "$API_PID_FILE"
  fi
  if [ -f "$UI_PID_FILE" ]; then
    kill "$(cat $UI_PID_FILE)" 2>/dev/null && echo "  ✓ UI dev server stopped"
    rm -f "$UI_PID_FILE"
  fi
  pkill -f "api_server.py" 2>/dev/null || true
  pkill -f "vite.*--host" 2>/dev/null || true
  exit 0
}

[ "$1" = "--stop" ] && stop_servers

echo "=== GraphRAG Servers ==="

# ── API server — use Python to detach with start_new_session ──
pkill -9 -f "api_server.py" 2>/dev/null || true
sleep 0.5
"$ROOT/.venv/bin/python3" -c "
import subprocess
proc = subprocess.Popen(
    ['$ROOT/.venv/bin/python', '-u', '$ROOT/api_server.py'],
    stdout=open('$API_LOG', 'w'),
    stderr=subprocess.STDOUT,
    cwd='$ROOT',
    start_new_session=True,
)
open('$API_PID_FILE', 'w').write(str(proc.pid))
print(proc.pid)
"

# Wait until it's up (max 15s)
for i in $(seq 1 15); do
  sleep 1
  if curl -sf http://localhost:5001/api/health > /dev/null 2>&1; then
    API_PID=$(cat "$API_PID_FILE" 2>/dev/null || echo "?")
    echo "  ✓ API server running  → http://localhost:5001  (PID $API_PID)"
    break
  fi
  if [ "$i" = "15" ]; then
    echo "  ✗ API server failed to start. Log:"
    tail -20 "$API_LOG"
    exit 1
  fi
done

# ── Vite UI ──
pkill -f "vite.*--host" 2>/dev/null || true
sleep 0.5
"$ROOT/.venv/bin/python3" -c "
import subprocess, os
proc = subprocess.Popen(
    ['npm', 'run', 'dev', '--', '--host', '0.0.0.0'],
    stdout=open('$UI_LOG', 'w'),
    stderr=subprocess.STDOUT,
    cwd='$ROOT/ui',
    start_new_session=True,
    env={**os.environ, 'PATH': os.environ.get('PATH', '') + ':/usr/bin:/usr/local/bin'},
)
open('$UI_PID_FILE', 'w').write(str(proc.pid))
print(proc.pid)
"

for i in $(seq 1 15); do
  sleep 1
  if curl -sf http://localhost:5173 > /dev/null 2>&1; then
    UI_PID=$(cat "$UI_PID_FILE" 2>/dev/null || echo "?")
    echo "  ✓ UI dev server running → http://localhost:5173  (PID $UI_PID)"
    break
  fi
  if [ "$i" = "15" ]; then
    echo "  ✗ UI dev server failed to start. Log:"
    tail -20 "$UI_LOG"
    exit 1
  fi
done

echo ""
echo "  Open → http://localhost:5173"
echo "  Stop → ./start_servers.sh --stop"
echo "  Logs → $API_LOG  |  $UI_LOG"
