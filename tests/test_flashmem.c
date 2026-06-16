/* FlashMemory retriever numerics parity test.
 *
 * Loads the converted DS4FMEM1 weights plus reference vectors produced by
 * tests/gen_flashmem_vectors.py and checks that the C query pipeline and
 * scorer reproduce the PyTorch reference logits.  Score logits decide chunk
 * recall by their sign, so the test checks both numeric closeness and the
 * recall decision agreement.
 *
 * Build (from the repo root):
 *     cc -O2 -o tests/test_flashmem tests/test_flashmem.c ds4_flashmem.c -lm -lpthread
 * Run:
 *     ./tests/test_flashmem flashmem-ds4.bin tests/test-vectors/flashmem-vectors.bin
 */

#include "../ds4_flashmem.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(FILE *fp, void *dst, size_t bytes) {
    return fread(dst, 1, bytes, fp) == bytes;
}

static int cmp_desc(const void *a, const void *b) {
    const float fa = *(const float *)a, fb = *(const float *)b;
    return (fa < fb) - (fa > fb);
}

static int test_kth_largest(void) {
    enum { N = 4097 };
    float *v = malloc(N * sizeof(float));
    float *scratch = malloc(N * sizeof(float));
    float *sorted = malloc(N * sizeof(float));
    uint64_t s = 88172645463325252ull;
    for (int i = 0; i < N; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        v[i] = (float)((int64_t)(s % 100000ull) - 50000) / 1000.0f;
    }
    memcpy(sorted, v, N * sizeof(float));
    qsort(sorted, N, sizeof(float), cmp_desc);
    int bad = 0;
    const uint32_t ks[] = { 1, 2, 512, 2048, N / 2, N - 1, N };
    for (size_t i = 0; i < sizeof(ks) / sizeof(ks[0]); i++) {
        const float got = ds4_flashmem_kth_largest(scratch, v, N, ks[i]);
        if (got != sorted[ks[i] - 1]) {
            fprintf(stderr, "kth_largest k=%u got %f want %f\n",
                    ks[i], got, sorted[ks[i] - 1]);
            bad = 1;
        }
    }
    free(sorted);
    free(scratch);
    free(v);
    return bad;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <flashmem-ds4.bin> <flashmem-vectors.bin>\n",
                argv[0]);
        return 1;
    }

    if (test_kth_largest()) {
        fprintf(stderr, "FAILED: kth_largest\n");
        return 1;
    }

    char err[256];
    ds4_flashmem *fm = ds4_flashmem_load(argv[1], err, sizeof(err));
    if (!fm) {
        fprintf(stderr, "FAILED: %s\n", err);
        return 1;
    }

    FILE *fp = fopen(argv[2], "rb");
    if (!fp) {
        fprintf(stderr, "FAILED: cannot open %s\n", argv[2]);
        return 1;
    }
    char magic[8];
    uint32_t n_cases, n_layers, n_embd, n_heads, head_dim, n_rows;
    if (!read_exact(fp, magic, 8) || memcmp(magic, "DS4FMVE1", 8) != 0 ||
        !read_exact(fp, &n_cases, 4) || !read_exact(fp, &n_layers, 4) ||
        !read_exact(fp, &n_embd, 4) || !read_exact(fp, &n_heads, 4) ||
        !read_exact(fp, &head_dim, 4) || !read_exact(fp, &n_rows, 4)) {
        fprintf(stderr, "FAILED: bad vectors header\n");
        return 1;
    }
    if (n_layers != fm->n_layers || n_embd != fm->n_embd ||
        n_heads != fm->n_heads || head_dim != fm->head_dim) {
        fprintf(stderr, "FAILED: vectors/weights shape mismatch\n");
        return 1;
    }

    float *hidden = malloc(n_embd * sizeof(float));
    float *keys = malloc((size_t)n_rows * head_dim * sizeof(float));
    float *expect = malloc((size_t)n_layers * n_rows * sizeof(float));
    float *q = malloc((size_t)n_heads * head_dim * sizeof(float));
    float *w = malloc(n_heads * sizeof(float));
    float *logits = malloc(n_rows * sizeof(float));

    double worst_abs = 0.0, worst_rel = 0.0;
    uint64_t decisions = 0, decision_mismatch = 0;
    int failed = 0;

    for (uint32_t c = 0; c < n_cases; c++) {
        uint32_t pos;
        if (!read_exact(fp, &pos, 4) ||
            !read_exact(fp, hidden, n_embd * sizeof(float)) ||
            !read_exact(fp, keys, (size_t)n_rows * head_dim * sizeof(float)) ||
            !read_exact(fp, expect, (size_t)n_layers * n_rows * sizeof(float))) {
            fprintf(stderr, "FAILED: truncated vectors at case %u\n", c);
            return 1;
        }
        for (uint32_t li = 0; li < n_layers; li++) {
            ds4_flashmem_build_query(fm, li, hidden, pos, q, w);
            ds4_flashmem_score_rows(fm, q, w, keys, n_rows, logits);
            const float *ref = expect + (size_t)li * n_rows;
            for (uint32_t r = 0; r < n_rows; r++) {
                const double abs_diff = fabs((double)logits[r] - ref[r]);
                const double rel =
                    abs_diff / (fabs((double)ref[r]) + 1e-3);
                if (abs_diff > worst_abs) worst_abs = abs_diff;
                if (rel > worst_rel) worst_rel = rel;
                decisions++;
                if ((logits[r] > 0.0f) != (ref[r] > 0.0f)) {
                    /* Sign flips are only acceptable within numeric noise of
                     * the decision boundary. */
                    if (fabs((double)ref[r]) > 5e-3) decision_mismatch++;
                }
            }
        }
        fprintf(stderr, "case %u pos=%u ok (worst abs so far %.3e)\n",
                c, pos, worst_abs);
    }

    /* The pipeline carries bf16 rounding and the f32 RoPE-angle quantization
     * of the reference table, so logits agree to a few 1e-2 absolute at the
     * largest positions; relative error is dominated by near-zero logits and
     * is reported but not gated.  The recall decision (logit sign) must agree
     * everywhere outside boundary noise. */
    if (worst_abs > 5e-2 || decision_mismatch != 0) failed = 1;

    fprintf(stderr,
            "flashmem parity: worst abs %.3e worst rel %.3e "
            "decision mismatches %llu/%llu -> %s\n",
            worst_abs, worst_rel,
            (unsigned long long)decision_mismatch,
            (unsigned long long)decisions,
            failed ? "FAILED" : "OK");
    return failed;
}
