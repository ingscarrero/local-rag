#!/usr/bin/env bash
# Stop the local model servers started by serve.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT/storage/pids"

for name in chat embed vlm; do
  f="$PID_DIR/$name.pid"
  if [ -f "$f" ]; then
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[$name] stopping pid $pid"; kill "$pid"
    fi
    rm -f "$f"
  fi
done
echo "Stopped."
