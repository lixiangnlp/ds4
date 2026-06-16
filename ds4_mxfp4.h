/*
 * Q4_K -> MXFP4 requantization for the Metal 4.1 TensorOps expert path.
 *
 * MXFP4 here is the OCP micro-scaling format that the Metal matmul2d blockwise
 * path consumes: a primary plane of 4-bit floats (metal_fp4_e2m1_format) plus an
 * auxiliary plane of per-32-element power-of-two scales (metal_fp8_ue8m0_format).
 *
 * Layout produced for one weight row of K elements (K a multiple of 32), matching
 * the Metal kernel in metal/ (data index = k + n*K for output column n; scale
 * index = (k/32) + n*(K/32)):
 *   - fp4 data : K/2 bytes. Element k is the low nibble of byte k/2 when k is
 *                even, the high nibble when k is odd.
 *   - ue8m0    : K/32 bytes, one power-of-two exponent per 32-element block.
 *
 * Requantization runs once at model load, so it uses a correct software
 * round-to-nearest-even quantizer rather than the hardware packer (whose
 * subnormal rounding is unreliable on macOS 27 beta).
 */
#ifndef DS4_MXFP4_H
#define DS4_MXFP4_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DS4_MXFP4_QK_K
#define DS4_MXFP4_QK_K 256
#endif

/* GGUF Q4_K superblock (256 elements). Mirrors the definition in ds4.c. */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[DS4_MXFP4_QK_K / 2];
} ds4_block_q4_K;

/*
 * Quantize 32 floats to one MXFP4 block: writes 16 packed nibble bytes
 * (nibbles_out[i] holds element 2*i in its low nibble and element 2*i+1 in its
 * high nibble) and returns the ue8m0 exponent byte for the block.
 */
uint8_t ds4_mxfp4_quant_block32(const float *x32, uint8_t *nibbles_out /*16 bytes*/);

/* Software reference decode: value of a 4-bit e2m1 nibble scaled by a ue8m0 byte.
 * Matches the hardware decode that matmul2d applies during the multiply. */
float ds4_mxfp4_dequant_element(unsigned nibble4, uint8_t ue8m0);

/*
 * Requantize one weight row of k floats (k a multiple of 32) to MXFP4.
 *   x        : k input weights.
 *   fp4_out  : k/2 bytes (packed nibbles).
 *   ue8m0_out: k/32 bytes (one scale per block).
 */
void ds4_mxfp4_requant_row(const float *x, int k, uint8_t *fp4_out, uint8_t *ue8m0_out);

/* Dequantize one Q4_K superblock row (k a multiple of 256) to k floats. */
void ds4_q4k_dequant_row(const ds4_block_q4_K *src, int k, float *out);

/*
 * Requantize one weight row stored as Q4_K superblocks (k a multiple of 256)
 * directly to MXFP4. Convenience wrapper around dequant + requant.
 */
void ds4_q4k_row_to_mxfp4(const ds4_block_q4_K *src, int k,
                          uint8_t *fp4_out, uint8_t *ue8m0_out);

#ifdef __cplusplus
}
#endif

#endif /* DS4_MXFP4_H */
