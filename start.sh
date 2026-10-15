#!/usr/bin/env bash
# YOLOFlow Studio - dev launcher (macOS / Linux / Git Bash)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ -d "$ROOT/frontend/node_modules" ] || (cd "$ROOT/frontend" && npm install)
(cd "$ROOT/backend" && python -m uvicorn main:app --host 127.0.0.1 --port 8000) &
(cd "$ROOT/frontend" && npm run dev) &
trap 'kill 0' EXIT
wait
