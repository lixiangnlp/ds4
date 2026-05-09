# ds4 benchmark harness

`ds4_bench.py` drives the checked-in `ds4` and `ds4-server` binaries and writes
one result directory per run under `bench/results/`.

Typical local comparison:

```sh
python3 bench/ds4_bench.py \
  --ctx 32768 \
  --tokens 32 \
  --prompt-chars 60000 \
  --run-kv \
  --timeout 7200 \
  --server-timeout 3600
```

Current M5 Max / 128 GB default recommendation is to run production service
with `DS4_METAL_PREFILL_CHUNK=4096` and a 32768 MiB disk KV budget. The GPU
top1 path is intentionally benchmark-only and is not enabled unless
`DS4_GREEDY_GPU_TOP1=1` is set explicitly.

The harness records:

- `prefill-chunk-4096` vs `prefill-chunk-8192` with `DS4_METAL_GRAPH_RAW_CAP=8192`.
- `decode-baseline-full-logits` vs `decode-gpu-top1` using `DS4_GREEDY_GPU_TOP1=1`.
- Optional `mtp-disabled` vs `mtp-enabled` when `--mtp path/to/mtp.gguf` is supplied.
- Optional disk-KV cold store and restart hit when `--run-kv` is supplied.

Each CLI case writes `*.json` with prompt tokens, generated tokens, token hash,
prefill/decode seconds, throughput, context-buffer estimate, and relevant env
values. Server KV cases keep `*.server.log` and parse cache miss, store, and hit
events into `results.json`.
