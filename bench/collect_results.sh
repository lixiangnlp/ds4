#!/bin/bash
# Aggregate baseline + optimized runs into bench/out/ab_results.json (consumed by
# ssd_streaming_viz.html) and print a comparison table.
set -u
ROOT=/Users/lixiang/project/ds4-origin
cd "$ROOT/bench/out"

read_csv(){ # $1 dir -> echoes "prefill_tps gen_tps kv_bytes" from last data row
  awk -F, 'NR>1 && $1!=""{p=$3;g=$5;k=$6} END{print p, g, k}' "$1/frontier.csv" 2>/dev/null
}
read_cache(){ grep -oE 'cached expert count: [0-9]+' "$1/bench.log" 2>/dev/null | grep -oE '[0-9]+' | head -1; }
read_peak(){ tail -n +2 "$1/samples.tsv" 2>/dev/null | awk -F'\t' '$3!="NA"&&$3!=""{v=$3} END{print v}'; }

read A_P A_G A_K <<<"$(read_csv baseline)"
read B_P B_G B_K <<<"$(read_csv optimized)"
A_EXP=$(read_cache baseline); B_EXP=$(read_cache optimized)
A_RAM=$(read_peak baseline);  B_RAM=$(read_peak optimized)

mb(){ awk -v b="$1" 'BEGIN{ if(b=="")print "—"; else printf "%.1f MB", b/1048576 }'; }
pct(){ awk -v a="$1" -v b="$2" 'BEGIN{ if(a==""||b==""||a+0==0){print "—"}else{printf "%+.1f%%",(b-a)/a*100} }'; }

cat > ab_results.json <<EOF
{
  "prefill_a":"${A_P:-—}", "prefill_b":"${B_P:-—}", "prefill_d":"$(pct "${A_P:-}" "${B_P:-}")",
  "decode_a":"${A_G:-—}",  "decode_b":"${B_G:-—}",  "decode_d":"$(pct "${A_G:-}" "${B_G:-}")",
  "kv_a":"$(mb "${A_K:-}")", "kv_b":"$(mb "${B_K:-}")", "kv_d":"$(pct "${A_K:-}" "${B_K:-}")",
  "ram_a":"${A_RAM:-—}",   "ram_b":"${B_RAM:-—}",   "ram_d":"peak phys_footprint",
  "exp_a":"${A_EXP:-—}",   "exp_b":"${B_EXP:-—}",   "exp_d":"$( [ "${A_EXP:-x}" = "${B_EXP:-y}" ] && echo '不变' || echo '变化' )"
}
EOF

echo "==================  A/B RESULTS  =================="
printf '%-22s %-16s %-16s %s\n' "metric" "baseline" "fp8-opt" "delta"
printf '%-22s %-16s %-16s %s\n' "prefill tok/s" "${A_P:-—}" "${B_P:-—}" "$(pct "${A_P:-}" "${B_P:-}")"
printf '%-22s %-16s %-16s %s\n' "decode tok/s"  "${A_G:-—}" "${B_G:-—}" "$(pct "${A_G:-}" "${B_G:-}")"
printf '%-22s %-16s %-16s %s\n' "KV cache"       "$(mb "${A_K:-}")" "$(mb "${B_K:-}")" "$(pct "${A_K:-}" "${B_K:-}")"
printf '%-22s %-16s %-16s %s\n' "peak phys RAM"  "${A_RAM:-—}" "${B_RAM:-—}" "(footprint)"
printf '%-22s %-16s %-16s %s\n' "cached experts" "${A_EXP:-—}" "${B_EXP:-—}" "$( [ "${A_EXP:-x}" = "${B_EXP:-y}" ] && echo same || echo diff )"
echo "wrote bench/out/ab_results.json"
