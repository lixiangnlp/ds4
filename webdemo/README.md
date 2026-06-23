# DS4 chat demo — DeepSeek-V4-Pro local inference

A ChatGPT-style multi-turn chat UI over the **ds4-server** inference backend
(DeepSeek-V4-Pro, 433 GB MoE) running the optimal **SSD-streaming + FP8-KV**
config, with a live panel showing prefill/decode/overall **TPS**, **TTFT**, token
counts, **RAM usage**, and **SSD throughput + cumulative I/O**.

![preview](preview.png)

## Run

```bash
# one command: starts ds4-server (optimal config) + the demo server, then open the URL
MODEL=~/Desktop/DeepSeek-V4-Pro-...-imatrix.gguf ./webdemo/start.sh
# -> open http://127.0.0.1:3000
```

The 433 GB model streams from SSD; load takes a few minutes. The status dot turns
green when `ds4-server` is up. (Default `MODEL` points at the Desktop GGUF.)

### Run the pieces separately
```bash
# backend (optimal config)
DS4_METAL_FP8_KV_STORE=1 ./ds4-server -m <model> --metal --ssd-streaming \
    --ctx 32768 --host 127.0.0.1 --port 8000 --cors

# demo server (UI + metrics + proxy)
PORT=3000 BACKEND=http://127.0.0.1:8000 node webdemo/server.js
```

## How the metrics are derived

- **Prefill / decode / overall TPS, TTFT** — computed in the browser from the SSE
  stream: TTFT = time to first content token; `prefill = prompt_tokens / TTFT`;
  `decode = completion_tokens / (t_last − t_first)`. Token counts come from the
  OpenAI `usage` chunk (`stream_options.include_usage`), falling back to counting
  streamed deltas + a char estimate.
- **RAM** — `footprint -p <ds4-server pid>` (phys_footprint + peak); the file-backed
  model mmap is excluded, so this is KV + activations + the resident expert cache.
- **SSD throughput + cumulative** — `iostat -d disk0` sampled once per second
  (read+write combined; under streaming this is read-dominated), integrated for the
  cumulative GB since the demo server started.

## Files
- `server.js` — zero-dep Node server: static UI, `/v1/chat/completions` proxy, `/api/metrics`.
- `index.html` — chat UI + live metrics dashboard (single file, no build).
- `start.sh` — launches backend + demo server with the optimal config.
