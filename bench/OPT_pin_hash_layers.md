# Optimization spec: pin hash-routed layers resident under SSD streaming

Status: **design complete + de-risked, core edit NOT landed** (see "Why not yet shipped").
Gating: `DS4_STREAM_PIN_HASH_LAYERS=1`, default off. Drift-safe by construction
(same weight bytes, only *where* experts live changes) — verify by output diff.

## Motivation (measured)

`bench/out/expert_profile.json` (DeepSeek-V4-Pro, 61 layers × 384 experts, top-6,
192 decode tokens):

- The first `DS4_N_HASH_LAYER` (=3) layers use **token-id hash routing**
  (`layer->ffn_gate_tid2eid != NULL`, ds4.c:7307). Selection is independent of the
  hidden state → ~uniform expert use (340/384 experts touched in 192 tokens),
  `avg_adjacent_jaccard ≈ 0.01`, per-layer n=64 cache hit ≈ **19%**.
- These layers are **uncacheable by popularity**: in the global slab cache they
  churn (constant evict/refill = SSD thrash) AND evict useful learned-layer experts.
- Fix: serve their experts from the **mapped (page-cache-resident) path** instead of
  the streaming slab cache. Cost to pin fully resident: 3 × 384 × 17.72 MiB ≈ **20 GiB**.

Hash layer test (static, no runtime detection): `il < DS4_N_HASH_LAYER`
(`DS4_N_HASH_LAYER` = `g_ds4_shape.n_hash_layer`, ds4.c:308).

## Implementation (3 confident pieces + 1 located-but-unlanded)

### A. Env gate — ds4.c (confident)
```c
static bool ds4_streaming_pin_hash_layers_enabled(void) {
    static int c = -1;
    if (c < 0) { const char *e = getenv("DS4_STREAM_PIN_HASH_LAYERS");
                 c = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return c;
}
static bool ds4_streaming_layer_pin_resident(uint32_t il) {
    return ds4_streaming_pin_hash_layers_enabled() && il < DS4_N_HASH_LAYER;
}
```

### B. Cover the mapped read — ds4.c:4282 `model_map_span_vec_include_layer_decode` (confident)
```c
    if (!weights_streaming_layer_experts_uniform(w, il) ||
        ds4_streaming_layer_pin_resident(il)) {            // <-- add
        model_map_span_vec_include_one(spans, l->ffn_gate_exps);
        model_map_span_vec_include_one(spans, l->ffn_up_exps);
        model_map_span_vec_include_one(spans, l->ffn_down_exps);
    }
```
This makes the hash layer's exps part of the always-mapped decode spans, so the
`wrap_model_range` mapped reads are covered (same prerequisite boosted layers use).

### C. Thread the flag — ds4_gpu.h + ds4.c (confident)
- ds4_gpu.h: add `uint8_t force_mapped;` to `struct ds4_gpu_stream_expert_table`.
- ds4.c `graph_stream_expert_table_make` (3373): set
  `table.force_mapped = ds4_streaming_layer_pin_resident(il);`
  (`il` is already a parameter; the table already carries `table.layer = il`.)

### D. Route force_mapped layers to the mapped matmul — ds4_metal.m (LOCATED MECHANISM, NOT LANDED)
The streaming decode classifies each layer as cache-resident vs mapped. Boosted /
off-size layers already take the mapped per-expert path because
`ds4_gpu_stream_expert_cache_note_expert_size` (ds4_metal.m:7563) *rejects* them
(returns 0 → that layer's experts read via `ds4_gpu_wrap_model_range` /
`encode_mul_mm_id_mapped`). Hash layers are the same byte-size, so they are NOT
rejected and currently go through the slab cache.

Required change: make the per-layer routed-MoE expert matmul take the **mapped
path** when `table->force_mapped` is set — i.e. treat it exactly like a
note_expert_size-rejected (budget-0) layer, but keyed on the flag rather than size.
The cache is per-layer indexed (`g_stream_expert_cache[layer][expert]`,
ds4_metal.m:710) and the per-(layer,expert) residency is available, so the hook is
feasible; the insertion point is the routed-expert matmul dispatch that chooses
slab-bound buffers vs `encode_mul_mm_id_mapped` (consumers of
`ds4_gpu_stream_expert_cache_begin_selected_load`, ds4.c:14378/14558/14621/14815/16193,
and the matmul in ds4_metal.m near the stream-expert buffer resolution 8279–10035).

NOTE: do NOT just leave hash-layer experts "uncached" — the cache *missing* path
streams them from SSD per token (thrash). They must take the **mapped-mm** path
(page-cache resident), the same one boosted layers use.

## Why not yet shipped

Piece D lands in the routed-MoE streaming expert matmul + global slab eviction —
the engine's most performance- and drift-sensitive kernel path. A wrong buffer bind
reads wrong weights (silent drift). Correct verification needs greedy-token output
diff (flag-on vs flag-off must be byte-identical) over real model runs (~10–40 min
each under 433 GB streaming). That is a focused, properly-tested PR, not a safe
in-session edit. Pieces A–C are safe scaffolding; D is the gated core change.

## Verification protocol (when landing D)
1. Build CPU + Metal clean (`make`).
2. `DS4_STREAM_PIN_HASH_LAYERS=0` and `=1`: `./ds4 -m <model> --ssd-streaming -p <fixed prompt> -n 64 --nothink`, diff the generated token sequences → MUST be identical (drift check).
3. `--expert-profile` before/after to confirm hash-layer SSD reads drop and overall
   decode tok/s improves; watch `g_stream_expert_cache_layer_*` and disk0 iostat.

## Safe interim — REFUTED by measurement (2026-06-22)

Growing the explicit resident slab was tried and **backfired hard** on this machine
(M5 Max 128 GiB):

| cache experts | peak phys | prefill tok/s | decode tok/s |
|--------------:|----------:|--------------:|-------------:|
| 3387 (auto 80%) | 74 GB | 54.70 | **0.32** |
| 4096 (explicit) | 86 GB | 15.21 | **0.02** (16× slower) |

Root cause: the explicit expert cache is wired/anonymous RAM that **competes with the
macOS unified buffer (OS page) cache**. SSD streaming reads the 433 GB mmap'd model
and depends on the page cache for its hot working set. The auto 80%-working-set budget
(3387) deliberately leaves ~20% for page cache + KV + activations. Pushing to 4096
dropped page-cache headroom from ~54 GB to ~42 GB, crossed the threshold where the
streamed hot set no longer fit, and every expert fetch hit physical SSD (plus likely
memory-compressor thrash) → decode collapsed.

**Corrected guidance:** do NOT raise `--ssd-streaming-cache-experts` above the auto
budget. FP8-freed RAM should be **left as OS page-cache headroom, not redirected into
the explicit cache** — that headroom is exactly why the FP8 run streams faster. The
auto 3387 is the sweet spot on this hardware. The real win for hash-layer churn is
piece D (mapped-resident hash layers), which does NOT enlarge the explicit slab.
