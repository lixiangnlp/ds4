/* FlashMemory lookahead retriever for DS4 ratio-4 compressed attention.
 * See ds4_flashmem.h for the role of this module.
 *
 * The query pipeline replicates the published PyTorch reference
 * (FlashMemory-Deepseek-V4 retriever.py) bit-for-bit where it matters:
 * f32 projections, f32 RMSNorm, YaRN RoPE over the trailing rope_dim values
 * with bf16 entry/exit rounding, then a normalized 128-wide Hadamard
 * transform.  Unlike the native lightning indexer there is no FP4
 * activation round-trip on the query side. */

#include "ds4_flashmem.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fm_fail(char *err, size_t err_len, const char *msg) {
    if (err && err_len) snprintf(err, err_len, "%s", msg);
}

/* ---------------------------------------------------------------------------
 * Small numeric helpers.
 * ------------------------------------------------------------------------- */

/* Round-to-nearest-even truncation to bfloat16, kept in float.  The reference
 * pipeline casts the query to bf16 before and after RoPE; skipping this shifts
 * scores near the recall threshold. */
static inline float fm_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r;
    memcpy(&r, &u, sizeof(r));
    return r;
}

/* Normalized in-place Walsh-Hadamard transform, natural (radix-2) order,
 * matching both the reference retriever and the DS4 indexer transform. */
static void fm_hadamard_inplace(float *x, uint32_t n) {
    for (uint32_t s = 1; s < n; s <<= 1) {
        for (uint32_t b = 0; b < n; b += 2u * s) {
            for (uint32_t i = 0; i < s; i++) {
                const float a = x[b + i];
                const float c = x[b + s + i];
                x[b + i] = a + c;
                x[b + s + i] = a - c;
            }
        }
    }
    const float inv = 1.0f / sqrtf((float)n);
    for (uint32_t i = 0; i < n; i++) x[i] *= inv;
}

/* ---------------------------------------------------------------------------
 * Threaded f32 matvec.  The wq_b projection is 16384x2048 and runs once per
 * retriever layer per trigger; a flat row split keeps the trigger hiccup at a
 * few milliseconds without dragging in the engine thread pool.
 * ------------------------------------------------------------------------- */

typedef struct {
    const float *w;
    const float *x;
    float *y;
    uint32_t cols;
    uint32_t row_beg;
    uint32_t row_end;
} fm_matvec_span;

static void *fm_matvec_worker(void *arg) {
    const fm_matvec_span *s = arg;
    for (uint32_t r = s->row_beg; r < s->row_end; r++) {
        const float *wr = s->w + (uint64_t)r * s->cols;
        float acc = 0.0f;
        for (uint32_t c = 0; c < s->cols; c++) acc += wr[c] * s->x[c];
        s->y[r] = acc;
    }
    return NULL;
}

static void fm_matvec(float *y, const float *w, const float *x,
                      uint32_t rows, uint32_t cols) {
    enum { FM_MATVEC_THREADS = 8 };
    if (rows < 1024) {
        fm_matvec_span span = { w, x, y, cols, 0, rows };
        fm_matvec_worker(&span);
        return;
    }
    pthread_t tid[FM_MATVEC_THREADS];
    fm_matvec_span span[FM_MATVEC_THREADS];
    const uint32_t step = (rows + FM_MATVEC_THREADS - 1) / FM_MATVEC_THREADS;
    uint32_t started = 0;
    for (uint32_t t = 0; t < FM_MATVEC_THREADS; t++) {
        const uint32_t beg = t * step;
        if (beg >= rows) break;
        uint32_t end = beg + step;
        if (end > rows) end = rows;
        span[t] = (fm_matvec_span){ w, x, y, cols, beg, end };
        if (pthread_create(&tid[t], NULL, fm_matvec_worker, &span[t]) != 0) {
            fm_matvec_worker(&span[t]);
            continue;
        }
        started = t + 1;
    }
    for (uint32_t t = 0; t < started; t++) pthread_join(tid[t], NULL);
}

/* ---------------------------------------------------------------------------
 * Loading.
 * ------------------------------------------------------------------------- */

static bool fm_read(FILE *fp, void *dst, size_t bytes) {
    return fread(dst, 1, bytes, fp) == bytes;
}

static float *fm_read_f32(FILE *fp, uint64_t count) {
    float *p = malloc(count * sizeof(float));
    if (!p) return NULL;
    if (!fm_read(fp, p, count * sizeof(float))) {
        free(p);
        return NULL;
    }
    return p;
}

/* YaRN frequency mixing as in the reference precompute_freqs_cis().  The
 * per-dimension frequencies are blended between the full-resolution and
 * factor-compressed bands; angles use these mixed frequencies directly. */
static float *fm_rope_freqs_build(const ds4_flashmem *fm) {
    const uint32_t half = fm->rope_dim / 2;
    float *mixed = malloc(half * sizeof(float));
    if (!mixed) return NULL;

    const double base = fm->rope_base;
    const double d = fm->rope_dim;
    const double max_pos = fm->rope_orig_len;
    const double low_f =
        d * log(max_pos / (fm->beta_fast * 2.0 * M_PI)) / (2.0 * log(base));
    const double high_f =
        d * log(max_pos / (fm->beta_slow * 2.0 * M_PI)) / (2.0 * log(base));
    double low = floor(low_f);
    double high = ceil(high_f);
    if (low < 0.0) low = 0.0;
    if (high > (double)half - 1.0) high = (double)half - 1.0;

    for (uint32_t i = 0; i < half; i++) {
        const double freq = 1.0 / pow(base, (double)(2 * i) / d);
        double ramp;
        if ((double)i < low) {
            ramp = 0.0;
        } else if ((double)i >= high) {
            ramp = 1.0;
        } else {
            const double span = high - low >= 1.0 ? high - low : 1.0;
            ramp = ((double)i - low) / span;
        }
        mixed[i] = (float)(freq * (1.0 - ramp) + freq / fm->rope_factor * ramp);
    }
    return mixed;
}

ds4_flashmem *ds4_flashmem_load(const char *path, char *err, size_t err_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fm_fail(err, err_len, "cannot open FlashMemory weights file");
        return NULL;
    }

    ds4_flashmem *fm = calloc(1, sizeof(*fm));
    char magic[8];
    bool ok = fm != NULL &&
              fm_read(fp, magic, sizeof(magic)) &&
              memcmp(magic, DS4_FLASHMEM_MAGIC, sizeof(magic)) == 0;
    if (ok) {
        ok = fm_read(fp, &fm->n_layers, sizeof(uint32_t)) &&
             fm_read(fp, &fm->n_embd, sizeof(uint32_t)) &&
             fm_read(fp, &fm->q_lora_rank, sizeof(uint32_t)) &&
             fm_read(fp, &fm->n_heads, sizeof(uint32_t)) &&
             fm_read(fp, &fm->head_dim, sizeof(uint32_t)) &&
             fm_read(fp, &fm->rope_dim, sizeof(uint32_t)) &&
             fm_read(fp, &fm->rope_base, sizeof(float)) &&
             fm_read(fp, &fm->rope_factor, sizeof(float)) &&
             fm_read(fp, &fm->rope_orig_len, sizeof(uint32_t)) &&
             fm_read(fp, &fm->beta_fast, sizeof(float)) &&
             fm_read(fp, &fm->beta_slow, sizeof(float)) &&
             fm_read(fp, &fm->rms_eps, sizeof(float));
    }
    if (ok && (fm->n_layers == 0 || fm->n_layers > DS4_FLASHMEM_MAX_LAYERS ||
               fm->n_embd == 0 || fm->q_lora_rank == 0 || fm->n_heads == 0 ||
               fm->head_dim == 0 || fm->rope_dim == 0 ||
               fm->rope_dim > fm->head_dim || fm->rope_dim > 128u ||
               (fm->rope_dim & 1u) ||
               (fm->head_dim & (fm->head_dim - 1u)) != 0)) {
        ok = false;
    }
    if (ok) {
        fm->score_scale =
            1.0f / sqrtf((float)fm->head_dim * (float)fm->n_heads);
        for (uint32_t li = 0; ok && li < fm->n_layers; li++) {
            ds4_flashmem_layer *l = &fm->layer[li];
            ok = fm_read(fp, &l->layer_id, sizeof(uint32_t)) &&
                 (l->wq_a = fm_read_f32(fp, (uint64_t)fm->q_lora_rank * fm->n_embd)) != NULL &&
                 (l->q_norm = fm_read_f32(fp, fm->q_lora_rank)) != NULL &&
                 (l->wq_b = fm_read_f32(fp, (uint64_t)fm->n_heads * fm->head_dim * fm->q_lora_rank)) != NULL &&
                 (l->wproj = fm_read_f32(fp, (uint64_t)fm->n_heads * fm->n_embd)) != NULL;
        }
    }
    if (ok) ok = (fm->rope_freqs = fm_rope_freqs_build(fm)) != NULL;

    fclose(fp);
    if (!ok) {
        fm_fail(err, err_len, "invalid or truncated FlashMemory weights file");
        ds4_flashmem_free(fm);
        return NULL;
    }
    return fm;
}

void ds4_flashmem_free(ds4_flashmem *fm) {
    if (!fm) return;
    for (uint32_t li = 0; li < DS4_FLASHMEM_MAX_LAYERS; li++) {
        free(fm->layer[li].wq_a);
        free(fm->layer[li].q_norm);
        free(fm->layer[li].wq_b);
        free(fm->layer[li].wproj);
    }
    free(fm->rope_freqs);
    free(fm);
}

/* ---------------------------------------------------------------------------
 * Query pipeline and scoring.
 * ------------------------------------------------------------------------- */

void ds4_flashmem_build_query(const ds4_flashmem *fm, uint32_t li,
                              const float *hidden, uint32_t pos,
                              float *q_out, float *w_out) {
    const ds4_flashmem_layer *l = &fm->layer[li];
    const uint32_t rank = fm->q_lora_rank;
    const uint32_t head_dim = fm->head_dim;
    const uint32_t rope_dim = fm->rope_dim;
    const uint32_t half = rope_dim / 2;

    float *lora = malloc(rank * sizeof(float));
    if (!lora) return;

    fm_matvec(lora, l->wq_a, hidden, rank, fm->n_embd);

    double ssq = 0.0;
    for (uint32_t i = 0; i < rank; i++) ssq += (double)lora[i] * lora[i];
    const float inv_norm =
        1.0f / sqrtf((float)(ssq / rank) + fm->rms_eps);
    for (uint32_t i = 0; i < rank; i++) lora[i] *= inv_norm * l->q_norm[i];

    fm_matvec(q_out, l->wq_b, lora, fm->n_heads * head_dim, rank);
    free(lora);

    /* The reference precomputes its RoPE table in f32: the angle is the f32
     * product pos * freq, and the retriever was trained with that
     * quantization baked in.  Reproduce the f32 product instead of computing
     * a more exact angle, or scores drift at large positions. */
    float rope_cos[64], rope_sin[64];
    for (uint32_t i = 0; i < half; i++) {
        const float angle = (float)pos * fm->rope_freqs[i];
        rope_cos[i] = cosf(angle);
        rope_sin[i] = sinf(angle);
    }
    for (uint32_t h = 0; h < fm->n_heads; h++) {
        float *qh = q_out + (uint64_t)h * head_dim;
        for (uint32_t i = 0; i < head_dim; i++) qh[i] = fm_bf16(qh[i]);
        float *qr = qh + (head_dim - rope_dim);
        for (uint32_t i = 0; i < half; i++) {
            const float c = rope_cos[i];
            const float s = rope_sin[i];
            const float a = qr[2 * i];
            const float b = qr[2 * i + 1];
            qr[2 * i] = fm_bf16(a * c - b * s);
            qr[2 * i + 1] = fm_bf16(a * s + b * c);
        }
        fm_hadamard_inplace(qh, head_dim);
    }

    fm_matvec(w_out, l->wproj, hidden, fm->n_heads, fm->n_embd);
}

float ds4_flashmem_kth_largest(float *scratch, const float *values,
                               uint32_t n, uint32_t k) {
    if (n == 0) return 0.0f;
    if (k == 0) k = 1;
    if (k > n) k = n;
    memcpy(scratch, values, (size_t)n * sizeof(float));

    /* Iterative quickselect for the k-th largest. */
    uint32_t lo = 0, hi = n - 1, want = k - 1;
    uint64_t seed = 0x9e3779b97f4a7c15ull ^ n;
    while (lo < hi) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        const float pivot = scratch[lo + (uint32_t)(seed % (hi - lo + 1u))];
        uint32_t i = lo, j = hi;
        while (i <= j) {
            while (scratch[i] > pivot) i++;
            while (scratch[j] < pivot) j--;
            if (i <= j) {
                const float tmp = scratch[i];
                scratch[i] = scratch[j];
                scratch[j] = tmp;
                i++;
                if (j == 0) break;
                j--;
            }
        }
        if (want <= j) {
            hi = j;
        } else if (want >= i) {
            lo = i;
        } else {
            return scratch[want];
        }
    }
    return scratch[want];
}

void ds4_flashmem_score_rows(const ds4_flashmem *fm, const float *q,
                             const float *w, const float *keys,
                             uint32_t n_rows, float *logits) {
    const uint32_t head_dim = fm->head_dim;
    for (uint32_t r = 0; r < n_rows; r++) {
        const float *key = keys + (uint64_t)r * head_dim;
        float acc = 0.0f;
        for (uint32_t h = 0; h < fm->n_heads; h++) {
            const float *qh = q + (uint64_t)h * head_dim;
            float dot = 0.0f;
            for (uint32_t i = 0; i < head_dim; i++) dot += qh[i] * key[i];
            if (dot > 0.0f) acc += dot * (w[h] * fm->score_scale);
        }
        logits[r] = acc;
    }
}
