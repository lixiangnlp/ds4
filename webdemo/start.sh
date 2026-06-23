#!/bin/bash
# Launch the DeepSeek-V4-Pro chat demo:
#   1. ds4-server with the optimal SSD-streaming + FP8-KV config (backend :8000)
#   2. the zero-dep Node demo server (UI + metrics + proxy, :3000)
#
# Model load over a 433 GB streaming GGUF takes a few minutes; the demo page
# shows "ds4-server 未运行" until the backend is up, then goes green.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-$HOME/Desktop/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf}"
PORT="${PORT:-3000}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
cd "$ROOT"

if [ ! -f "$MODEL" ]; then echo "model not found: $MODEL (set MODEL=...)"; exit 1; fi

echo "[demo] starting ds4-server (optimal config) on :$BACKEND_PORT ..."
DS4_METAL_FP8_KV_STORE=1 ./ds4-server \
    -m "$MODEL" --metal --ssd-streaming --ctx 32768 \
    --host 127.0.0.1 --port "$BACKEND_PORT" --cors \
    > webdemo/ds4-server.log 2>&1 &
SRV=$!
echo "[demo] ds4-server pid=$SRV (loading model, see webdemo/ds4-server.log)"

cleanup(){ echo; echo "[demo] stopping..."; kill "$SRV" 2>/dev/null; kill "$NODE" 2>/dev/null; exit 0; }
trap cleanup INT TERM

PORT="$PORT" BACKEND="http://127.0.0.1:$BACKEND_PORT" node webdemo/server.js &
NODE=$!
echo "[demo] open  http://127.0.0.1:$PORT"
wait "$NODE"
