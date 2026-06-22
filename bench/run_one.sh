#!/bin/bash
# Run one ds4-bench SSD-streaming config, sampling peak RAM (footprint) and
# disk0 throughput (iostat) throughout.  Args:
#   $1 = label (dir under bench/out)
#   $2 = env assignment, e.g. "DS4_DISABLE_KV_OPTS=1" or "DS4_METAL_FP8_KV_STORE=1"
# Optional env overrides: CTXF (frontier), GEN (gen tokens).
set -u
ROOT=/Users/lixiang/project/ds4-origin
MODEL="$HOME/Desktop/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
PROMPT="$ROOT/bench/merged_prompt.txt"
LABEL="$1"
ENVASSIGN="$2"
CTXF="${CTXF:-32768}"
GEN="${GEN:-1024}"

OUT="$ROOT/bench/out/$LABEL"
mkdir -p "$OUT"
LOG="$OUT/bench.log"
SAMP="$OUT/samples.tsv"
CSV="$OUT/frontier.csv"
printf 'epoch\tphys_cur\tphys_peak\tdisk_mbps\n' > "$SAMP"

echo "[$(date '+%H:%M:%S')] launching $LABEL ($ENVASSIGN) frontier=$CTXF gen=$GEN"
cd "$ROOT"
env "$ENVASSIGN" ./ds4-bench -m "$MODEL" --metal --ssd-streaming \
    --prompt-file "$PROMPT" --ctx-start "$CTXF" --ctx-max "$CTXF" \
    --gen-tokens "$GEN" --csv "$CSV" > "$LOG" 2>&1 &
PID=$!
echo "[$(date '+%H:%M:%S')] $LABEL pid=$PID"

# Sample until the bench process exits.  footprint is a cheap kernel query even
# for a 433GB-mmap process; iostat -c 2 -w 1 yields a 1s interval reading.
while kill -0 "$PID" 2>/dev/null; do
    FP=$(footprint -p "$PID" 2>/dev/null)
    cur=$(printf '%s\n' "$FP" | awk '/phys_footprint:/{print $2$3}')
    peak=$(printf '%s\n' "$FP" | awk '/phys_footprint_peak:/{print $2$3}')
    dmbps=$(iostat -d -c 2 -w 1 disk0 2>/dev/null | tail -1 | awk '{print $3}')
    printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "${cur:-NA}" "${peak:-NA}" "${dmbps:-NA}" >> "$SAMP"
done
wait "$PID"; RC=$?
echo "[$(date '+%H:%M:%S')] $LABEL done rc=$RC"
echo "PEAK_RAM=$(tail -n +2 "$SAMP" | awk -F'\t' '{print $3}' | tail -1)"
exit $RC
