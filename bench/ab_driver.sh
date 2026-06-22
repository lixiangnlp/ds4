#!/bin/bash
# A/B driver: baseline (original KV layout) vs optimized (packed FP8 KV),
# both under SSD streaming at the same 32768 frontier, with a >=5 min thermal
# gap between runs so SSD/GPU heat does not bias the second run.
set -u
ROOT=/Users/lixiang/project/ds4-origin
cd "$ROOT"
export CTXF=32768
export GEN=256

echo "[$(date '+%H:%M:%S')] === A/B START (frontier=$CTXF gen=$GEN) ==="
bash bench/run_one.sh baseline  "DS4_DISABLE_KV_OPTS=1"
echo "[$(date '+%H:%M:%S')] baseline finished; cooling down 330s for thermal parity"
sleep 330
bash bench/run_one.sh optimized "DS4_METAL_FP8_KV_STORE=1"
echo "[$(date '+%H:%M:%S')] === A/B COMPLETE ==="
