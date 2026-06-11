#!/usr/bin/env bash
# Start the three local model servers (llama.cpp `llama-server`, OpenAI-compatible).
# Each model is its own process so it loads once and stays warm.
#
#   chat LLM   -> :8080   (agent reasoning, grading, generation)
#   embeddings -> :8081   (text chunks + figure captions)
#   vision VLM -> :8082   (figure captioning during ingestion)
#
# Models are auto-downloaded from Hugging Face on first run (GGUF, cached locally).
# Override any model with env vars before running, e.g.:
#   CHAT_HF=bartowski/Qwen2.5-14B-Instruct-GGUF:Q4_K_M ./scripts/serve.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/storage/logs"
PID_DIR="$ROOT/storage/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# Default models. The M3 Ultra / 512GB can comfortably run far larger — bump these.
CHAT_HF="${CHAT_HF:-bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M}"
EMBED_HF="${EMBED_HF:-nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M}"
VLM_HF="${VLM_HF:-ggml-org/Qwen2.5-VL-7B-Instruct-GGUF}"

NGL="${NGL:-99}"   # offload all layers to Metal

start() {
  local name="$1"; shift
  local port="$1"; shift
  if [ -f "$PID_DIR/$name.pid" ] && kill -0 "$(cat "$PID_DIR/$name.pid")" 2>/dev/null; then
    echo "[$name] already running on :$port (pid $(cat "$PID_DIR/$name.pid"))"
    return
  fi
  echo "[$name] starting on :$port …  (log: $LOG_DIR/$name.log)"
  nohup "$@" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$PID_DIR/$name.pid"
}

start chat 8080 \
  llama-server -hf "$CHAT_HF" --port 8080 -ngl "$NGL" -c 8192 --jinja

start embed 8081 \
  llama-server -hf "$EMBED_HF" --port 8081 -ngl "$NGL" --embedding --pooling mean -ub 2048

start vlm 8082 \
  llama-server -hf "$VLM_HF" --port 8082 -ngl "$NGL" -c 8192

echo
echo "Waiting for servers to become healthy …"
for p in 8080 8081 8082; do
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1; then
      echo "  :$p healthy"; break
    fi
    sleep 2
  done
done
echo "Done. Tail logs with: tail -f $LOG_DIR/*.log"
