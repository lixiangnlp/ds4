#ifndef DS4_FLASHMEM_H
#define DS4_FLASHMEM_H

/* FlashMemory lookahead retriever (arXiv 2606.09079) for DS4 ratio-4
 * compressed attention.  Given the attention-norm hidden state of a decode
 * token, the retriever predicts which compressed CSA chunks the next ~64
 * tokens will attend to, so the native lightning indexer only has to scan the
 * resident subset.  This module owns the retriever weights and the exact
 * query pipeline of the published checkpoint; scoring against the compressed
 * indexer keys reuses the same relu-dot-weighted-sum form as the native
 * indexer, so the Metal path can share that kernel.
 *
 * The module is self-contained (libc + math only) so the numerics can be
 * regression-tested against the PyTorch reference without the engine. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_FLASHMEM_MAGIC "DS4FMEM1"
#define DS4_FLASHMEM_MAX_LAYERS 8

/* Gating policy: re-score every tau decode steps and keep the top-scoring
 * chunks by the cross-layer max ensemble, plus the chunks covering the most
 * recent 8K tokens (2048 ratio-4 chunks) unconditionally.  Top-k selection
 * (the released checkpoint's demo default) is used instead of the paper's
 * absolute sigmoid threshold: the published weights ship without the
 * threshold-fallback mechanism and their logit calibration keeps nearly
 * everything at 0.5, while their ranking is informative.  KEEP_SHIFT 3 keeps
 * 1/8th of the history, matching the paper's ~13.5% average residency, with
 * a floor so small histories are never gated hard. */
enum {
    DS4_FLASHMEM_TAU = 64,
    DS4_FLASHMEM_RECENT_CHUNKS = 2048,
    DS4_FLASHMEM_MIN_CHUNKS = 4096,
    DS4_FLASHMEM_KEEP_SHIFT = 3,
    DS4_FLASHMEM_KEEP_FLOOR = 2048,
};

typedef struct {
    uint32_t layer_id;   /* model layer index this retriever was trained for */
    float *wq_a;         /* [q_lora_rank, n_embd] */
    float *q_norm;       /* [q_lora_rank] */
    float *wq_b;         /* [n_heads * head_dim, q_lora_rank] */
    float *wproj;        /* [n_heads, n_embd] */
} ds4_flashmem_layer;

typedef struct {
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t q_lora_rank;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t rope_dim;
    float rope_base;
    float rope_factor;
    uint32_t rope_orig_len;
    float beta_fast;
    float beta_slow;
    float rms_eps;
    float score_scale;   /* head_dim^-0.5 * n_heads^-0.5, applied like the
                          * native indexer scale */
    ds4_flashmem_layer layer[DS4_FLASHMEM_MAX_LAYERS];
    float *rope_freqs;   /* [rope_dim / 2] YaRN-mixed angular frequencies */
} ds4_flashmem;

ds4_flashmem *ds4_flashmem_load(const char *path, char *err, size_t err_len);
void ds4_flashmem_free(ds4_flashmem *fm);

/* Build the lookahead query for retriever layer index li (0..n_layers-1) from
 * the n_embd-wide hidden state of the decode token at position pos.
 * q_out: n_heads * head_dim floats.  w_out: n_heads floats (un-scaled head
 * weights; combine with fm->score_scale exactly like the native indexer). */
void ds4_flashmem_build_query(const ds4_flashmem *fm, uint32_t li,
                              const float *hidden, uint32_t pos,
                              float *q_out, float *w_out);

/* CPU reference scorer: logits[r] = sum_h relu(q_h . key_r) * w_h * scale
 * over n_rows keys of head_dim floats. */
void ds4_flashmem_score_rows(const ds4_flashmem *fm, const float *q,
                             const float *w, const float *keys,
                             uint32_t n_rows, float *logits);

/* k-th largest of values[0..n) (k >= 1), used as the top-k keep cutoff.
 * scratch must hold n floats; values are left untouched. */
float ds4_flashmem_kth_largest(float *scratch, const float *values,
                               uint32_t n, uint32_t k);

#endif
