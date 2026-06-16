#include "ds4_mxfp4.h"

#include <math.h>
#include <string.h>

/* e2m1 magnitudes in 3-bit code order: code = (exp<<1)|mantissa.
 * 0:0.0 1:0.5 2:1.0 3:1.5 4:2.0 5:3.0 6:4.0 7:6.0  (bit3 carries the sign). */
static const float DS4_E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

/* Round a non-negative magnitude to the nearest e2m1 code, ties to even code. */
static int ds4_e2m1_round(float ax) {
    int best = 0;
    float best_diff = fabsf(ax - DS4_E2M1[0]);
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - DS4_E2M1[i]);
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return best;
}

static inline float ds4_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const int32_t exp = (h >> 10) & 0x1F;
    const uint32_t frac = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        bits = sign; /* flush subnormals to signed zero; matches ds4 dequant use */
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (frac << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (frac << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint8_t ds4_mxfp4_quant_block32(const float *x32, uint8_t *nibbles_out) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        const float a = fabsf(x32[i]);
        if (a > amax) amax = a;
    }

    int exp;
    if (amax <= 0.0f || !isfinite(amax)) {
        /* All-zero (or degenerate) block: zero nibbles, neutral scale. */
        memset(nibbles_out, 0, 16);
        return 127; /* 2^0 */
    }
    /* OCP MX shared scale: align the block max to the e2m1 max-normal exponent
     * (6.0 = 1.5 * 2^2), so exp = floor(log2(amax)) - 2. */
    exp = (int)floorf(log2f(amax)) - 2;
    if (exp < -127) exp = -127;
    if (exp > 127) exp = 127;
    const float scale = ldexpf(1.0f, exp);
    const float inv_scale = 1.0f / scale;

    memset(nibbles_out, 0, 16);
    for (int i = 0; i < 32; i++) {
        const float v = x32[i];
        const unsigned sign = (v < 0.0f) ? 0x8u : 0x0u;
        const int code = ds4_e2m1_round(fabsf(v) * inv_scale);
        const unsigned nib = sign | (unsigned)code;
        if (i & 1) {
            nibbles_out[i >> 1] |= (uint8_t)(nib << 4);
        } else {
            nibbles_out[i >> 1] |= (uint8_t)nib;
        }
    }
    return (uint8_t)(exp + 127);
}

float ds4_mxfp4_dequant_element(unsigned nibble4, uint8_t ue8m0) {
    const float mag = DS4_E2M1[nibble4 & 0x7];
    const float sign = (nibble4 & 0x8) ? -1.0f : 1.0f;
    const float scale = ldexpf(1.0f, (int)ue8m0 - 127);
    return sign * mag * scale;
}

void ds4_mxfp4_requant_row(const float *x, int k, uint8_t *fp4_out, uint8_t *ue8m0_out) {
    const int nblocks = k / 32;
    for (int b = 0; b < nblocks; b++) {
        ue8m0_out[b] = ds4_mxfp4_quant_block32(x + b * 32, fp4_out + b * 16);
    }
}

void ds4_q4k_dequant_row(const ds4_block_q4_K *src, int k, float *out) {
    const int nb = k / DS4_MXFP4_QK_K;
    for (int i = 0; i < nb; i++) {
        const ds4_block_q4_K *bx = &src[i];
        const float d = ds4_f16_to_f32(bx->d);
        const float dm = ds4_f16_to_f32(bx->dmin);
        float *dst = out + (size_t)i * DS4_MXFP4_QK_K;

        for (int j = 0; j < DS4_MXFP4_QK_K / 32; j++) {
            uint8_t sc, m;
            /* Q4_K packs eight 6-bit (scale, min) pairs across 12 bytes. */
            if (j < 4) {
                sc = bx->scales[j] & 63;
                m  = bx->scales[j + 4] & 63;
            } else {
                sc = (bx->scales[j + 4] & 0xF) | ((bx->scales[j - 4] >> 6) << 4);
                m  = (bx->scales[j + 4] >> 4)  | ((bx->scales[j - 0] >> 6) << 4);
            }
            const int byte_off = (j >> 1) * 32;
            const int shift = (j & 1) * 4;
            for (int l = 0; l < 32; l++) {
                const int q = (bx->qs[byte_off + l] >> shift) & 0xF;
                dst[j * 32 + l] = d * (float)sc * (float)q - dm * (float)m;
            }
        }
    }
}

void ds4_q4k_row_to_mxfp4(const ds4_block_q4_K *src, int k,
                          uint8_t *fp4_out, uint8_t *ue8m0_out) {
    /* Dequantize in 256-element chunks to bound the temporary, then requant. */
    float tmp[DS4_MXFP4_QK_K];
    const int nb = k / DS4_MXFP4_QK_K;
    for (int i = 0; i < nb; i++) {
        ds4_q4k_dequant_row(&src[i], DS4_MXFP4_QK_K, tmp);
        /* Each 256-chunk is 8 MXFP4 blocks of 32: 8*16 fp4 bytes, 8 scales. */
        ds4_mxfp4_requant_row(tmp, DS4_MXFP4_QK_K,
                              fp4_out + (size_t)i * (DS4_MXFP4_QK_K / 2),
                              ue8m0_out + (size_t)i * (DS4_MXFP4_QK_K / 32));
    }
}
