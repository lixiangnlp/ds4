/*
 * Unit tests for the Q4_K -> MXFP4 requantizer (ds4_mxfp4.c).
 * Build: cc -O2 -Wall -Wextra -std=c99 -I. -o tests/test_mxfp4 tests/test_mxfp4.c ds4_mxfp4.c -lm
 * Run:   ./tests/test_mxfp4
 *
 * The Metal matmul2d MXFP4 path is validated separately on-GPU; these tests
 * pin down the host-side requantization: the e2m1/ue8m0 encoding, the packed
 * nibble layout the kernel reads, and the precision cost relative to Q4_K.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../ds4_mxfp4.h"

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", (msg)); g_fail = 1; } \
} while (0)

/* The exact e2m1 grid the encoder targets. */
static const float E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

/* ---- 1. block quantizer: encoding, layout, and grid-snapping ---- */
static void test_block_quant(void) {
    printf("test: block quantizer encoding + layout\n");

    /* Exactly representable values at scale 2^0 should round-trip exactly. */
    float x[32];
    for (int i = 0; i < 32; i++) x[i] = (i & 1 ? -1.0f : 1.0f) * E2M1[i & 7];
    uint8_t nib[16];
    uint8_t s = ds4_mxfp4_quant_block32(x, nib);
    CHECK(s == 127, "scale for max-6.0 block should be 2^0 (ue8m0=127)");

    for (int i = 0; i < 32; i++) {
        unsigned code = (i & 1) ? (nib[i >> 1] >> 4) : (nib[i >> 1] & 0xF);
        float got = ds4_mxfp4_dequant_element(code, s);
        CHECK(got == x[i], "exactly representable element must round-trip");
        /* low/high nibble placement */
        CHECK((code & 0x7) == (unsigned)(i & 7), "magnitude code placement");
        CHECK(((code & 0x8) != 0) == ((i & 1) != 0), "sign bit placement");
    }

    /* Scale selection: a block whose max is 12.0 needs exp=1 (ue8m0=128) so
     * 12.0/2 = 6.0 lands on the e2m1 max. */
    for (int i = 0; i < 32; i++) x[i] = (i == 0) ? 12.0f : 0.0f;
    s = ds4_mxfp4_quant_block32(x, nib);
    CHECK(s == 128, "block max 12.0 -> ue8m0=128 (2^1)");
    CHECK(ds4_mxfp4_dequant_element(nib[0] & 0xF, s) == 12.0f, "12.0 reconstructs exactly");

    /* All-zero block: neutral scale, zero nibbles. */
    memset(x, 0, sizeof(x));
    s = ds4_mxfp4_quant_block32(x, nib);
    CHECK(s == 127, "zero block -> ue8m0=127");
    int allzero = 1;
    for (int i = 0; i < 16; i++) if (nib[i]) allzero = 0;
    CHECK(allzero, "zero block -> zero nibbles");

    /* Round-to-nearest reconstruction: a uniform 0.7 block picks the block
     * scale from amax=0.7 (exp=-3, scale=0.125) and snaps each element to the
     * nearest grid point (0.75), so reconstruction error stays within one step. */
    for (int i = 0; i < 32; i++) x[i] = 0.7f;
    s = ds4_mxfp4_quant_block32(x, nib);
    float r = ds4_mxfp4_dequant_element(nib[0] & 0xF, s);
    CHECK(fabsf(r - 0.7f) <= 0.0625f + 1e-6f, "0.7 reconstructs within half a grid step");
}

/* ---- 2. requant row layout matches the kernel's index math ---- */
static void test_row_layout(void) {
    printf("test: row requant layout (data idx=k, scale idx=k/32)\n");
    const int K = 128; /* 4 blocks */
    float x[128];
    for (int k = 0; k < K; k++) x[k] = sinf(k * 0.3f) * 3.0f;

    uint8_t fp4[128 / 2];
    uint8_t scales[128 / 32];
    ds4_mxfp4_requant_row(x, K, fp4, scales);

    /* Decode element k from the packed row and compare to a fresh per-block
     * quantization to confirm the row writer places nibbles/scales correctly. */
    double max_block_mismatch = 0.0;
    for (int b = 0; b < K / 32; b++) {
        uint8_t nib[16];
        uint8_t s = ds4_mxfp4_quant_block32(x + b * 32, nib);
        CHECK(s == scales[b], "row scale matches standalone block scale");
        for (int i = 0; i < 32; i++) {
            int k = b * 32 + i;
            unsigned code_row = (k & 1) ? (fp4[k >> 1] >> 4) : (fp4[k >> 1] & 0xF);
            unsigned code_blk = (i & 1) ? (nib[i >> 1] >> 4) : (nib[i >> 1] & 0xF);
            if (code_row != code_blk) max_block_mismatch = 1.0;
        }
    }
    CHECK(max_block_mismatch == 0.0, "row nibble layout matches per-block layout");
}

/* ---- 3. Q4_K -> MXFP4 precision vs the original Q4_K dequant ---- */
static void fill_q4k(ds4_block_q4_K *bx, uint32_t seed) {
    uint8_t *p = (uint8_t *)bx;
    uint32_t s = seed;
    for (size_t i = 0; i < sizeof(*bx); i++) { s = s * 1664525u + 1013904223u; p[i] = (uint8_t)(s >> 24); }
    /* Keep d, dmin as modest positive f16 values so dequant is well-scaled. */
    bx->d = 0x3000;    /* ~0.125 */
    bx->dmin = 0x2800; /* ~0.03 */
}

static void test_q4k_precision(void) {
    printf("test: Q4_K -> MXFP4 precision\n");
    const int NB = 16;                  /* superblocks per row */
    const int K = NB * DS4_MXFP4_QK_K;  /* 4096 */
    ds4_block_q4_K *row = malloc((size_t)NB * sizeof(ds4_block_q4_K));
    float *ref = malloc((size_t)K * sizeof(float));
    uint8_t *fp4 = malloc((size_t)K / 2);
    uint8_t *scales = malloc((size_t)K / 32);

    for (int i = 0; i < NB; i++) fill_q4k(&row[i], 0xC0FFEEu + i * 7u);
    ds4_q4k_dequant_row(row, K, ref);
    ds4_q4k_row_to_mxfp4(row, K, fp4, scales);

    /* Reconstruct from MXFP4 and measure error vs the Q4_K dequant. Also do a
     * dot product against a random activation both ways (this is exactly what
     * the GPU matmul computes) to confirm layout + decode self-consistency. */
    double sse = 0, sref = 0, max_rel = 0;
    double dot_ref = 0, dot_mx = 0;
    uint32_t rng = 12345;
    for (int k = 0; k < K; k++) {
        unsigned code = (k & 1) ? (fp4[k >> 1] >> 4) : (fp4[k >> 1] & 0xF);
        double mx = ds4_mxfp4_dequant_element(code, scales[k >> 5]);
        double e = mx - ref[k];
        sse += e * e; sref += (double)ref[k] * ref[k];
        double denom = fabs(ref[k]) + 1e-4;
        double rel = fabs(e) / denom;
        if (rel > max_rel) max_rel = rel;

        rng = rng * 1664525u + 1013904223u;
        double a = ((double)(rng >> 8) / 16777216.0) * 2.0 - 1.0;
        dot_ref += a * ref[k];
        dot_mx  += a * mx;
    }
    double rms = sqrt(sse / K);
    double rel_l2 = sqrt(sse / (sref + 1e-12));
    printf("  K=%d  RMS_err=%.5g  rel_L2=%.4f  max_elem_rel=%.3f\n", K, rms, rel_l2, max_rel);
    printf("  dot: ref=%.5f  mxfp4=%.5f  rel_diff=%.4f\n",
           dot_ref, dot_mx, fabs(dot_ref - dot_mx) / (fabs(dot_ref) + 1e-6));

    /* MXFP4 over a 4-bit Q4_K source: the relative L2 error should be modest
     * (single-digit percent), confirming the requant is sane rather than
     * scrambled. This is a sanity bound, not a quality verdict. */
    CHECK(rel_l2 < 0.15, "MXFP4 relative-L2 error vs Q4_K should be < 15%");
    CHECK(isfinite(rms) && rms >= 0.0, "RMS finite");

    free(row); free(ref); free(fp4); free(scales);
}

int main(void) {
    printf("ds4 MXFP4 requantizer tests\n");
    test_block_quant();
    test_row_layout();
    test_q4k_precision();
    if (g_fail) { printf("mxfp4 tests: FAIL\n"); return 1; }
    printf("mxfp4 tests: ok\n");
    return 0;
}
