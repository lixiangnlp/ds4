#!/usr/bin/env python3
"""Generate FlashMemory retriever parity vectors for tests/test_flashmem.c.

Runs random hidden states and fp8-compressed keys through the published
PyTorch reference (retriever.py from the FlashMemory-Deepseek-V4 release) and
dumps inputs plus expected per-layer logits as a flat binary.

Usage:
    python3 tests/gen_flashmem_vectors.py \
        --repo /Users/lixiang/project/FlashMemory-Deepseek-V4/repo \
        --out tests/test-vectors/flashmem-vectors.bin

Output (little-endian):
    char[8] "DS4FMVE1"
    u32 n_cases, n_layers, n_embd, n_heads, head_dim, n_rows
    per case:
        u32 position
        f32 hidden [n_embd]
        f32 keys   [n_rows * head_dim]      (fp8-dequantized, what ds4 scores)
        f32 logits [n_layers * n_rows]      (reference, pre-sigmoid)
"""

import argparse
import struct
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--rows", type=int, default=512)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from retriever import FlashMemoryRetriever, dequant_compressed_k

    torch.manual_seed(20260611)
    model = FlashMemoryRetriever.from_checkpoint(
        f"{args.repo}/weights/flashmemory_ds_v4.safetensors",
        device="cpu", max_position=1048576)

    n_embd = 4096
    head_dim = model.head_dim
    n_rows = args.rows
    positions = [0, 63, 4096, 65535, 131072, 524287][: args.cases]

    with open(args.out, "wb") as f:
        f.write(b"DS4FMVE1")
        f.write(struct.pack("<6I", len(positions), len(model.layer_names),
                            n_embd, model.n_heads, head_dim, n_rows))
        for pos in positions:
            hidden = torch.randn(1, n_embd) * 0.5
            comp = torch.randint(0, 256, (1, n_rows, head_dim + 4),
                                 dtype=torch.uint8)
            # Rewrite the scale bytes with sane positive f32 scales so the
            # dequantized keys look like real compressed rows.
            scales = (torch.rand(n_rows) * 0.05 + 0.01).numpy().astype("<f4")
            comp_np = comp.numpy()
            comp_np[0, :, head_dim:] = np.frombuffer(
                scales.tobytes(), dtype=np.uint8).reshape(n_rows, 4)
            comp = torch.from_numpy(comp_np)

            keys = dequant_compressed_k(comp, head_dim=head_dim)  # [1,N,D] f32
            keys = torch.nan_to_num(keys, nan=0.0)
            per_layer = model.forward(hidden, comp, torch.tensor([pos]),
                                      apply_sigmoid=False)
            # NaN fp8 payloads (0x7f/0xff) were zeroed in `keys`; rescore the
            # reference on the cleaned keys so both sides see identical input.
            logits = [model.scorers[name].logits(hidden, keys,
                                                 torch.tensor([pos]),
                                                 model.freqs_cis)
                      for name in model.layer_names]
            del per_layer

            f.write(struct.pack("<I", pos))
            f.write(hidden.numpy().astype("<f4").tobytes())
            f.write(keys.numpy().astype("<f4").tobytes())
            for lg in logits:
                f.write(lg.numpy().astype("<f4").tobytes())
    print(f"wrote {args.out}: cases={positions} rows={n_rows}")


if __name__ == "__main__":
    main()
