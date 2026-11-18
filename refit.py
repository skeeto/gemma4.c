#!/usr/bin/env python3
"""Rewrite the q4_0 tensors of Google's E2B GGUF from the QAT checkpoint.

Google's quantization-aware checkpoint (gemma-4-E2B-it-qat-q4_0-unquantized)
holds weights that lie on a 4-bit grid: each block of 32 is a step times
integers in -8..7, stored in bf16. Its q4_0 GGUF quantizes them again with
llama.cpp's rule, a step of the block's largest-magnitude value over -8,
which misses the trained steps by about 5%. This fits each block's own step
instead and writes a copy of the GGUF with its q4_0 data replaced, block for
block, leaving everything else (metadata, q6_K embeddings, norms) as it is.

    python3 refit.py gemma-4-E2B_q4_0-it.gguf \\
        gemma-4-E2B-it-qat-q4_0-unquantized/model.safetensors \\
        -o gemma-4-E2B-it-qat-q4_0-refit.gguf

With --check it fits the matrices of a checkpoint and reports the residual
without writing anything.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys

import numpy as np

GGML_Q4_0 = 2
GGUF_TO_HF = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj", "inp_gate": "per_layer_input_gate", "proj": "per_layer_projection",
}


class SafeTensors:
    """Reads bf16 tensors from a safetensors file without loading it all."""

    def __init__(self, path):
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(size))
        self.data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size)

    def get(self, name):
        info = self.header[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name} is {info['dtype']}, expected BF16")
        start, end = info["data_offsets"]
        raw = self.data[start:end].view(np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])


def read_gguf_tensors(path):
    """Returns {name: (dims, type, absolute offset)} from a GGUF header."""
    with open(path, "rb") as f:
        head = f.read(64 << 20)
    pos = 0

    def take(fmt):
        nonlocal pos
        value = struct.unpack_from("<" + fmt, head, pos)
        pos += struct.calcsize("<" + fmt)
        return value[0]

    def text():
        nonlocal pos
        n = take("Q")
        pos += n
        return head[pos - n:pos].decode()

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(t):
        nonlocal pos
        if t == 8:
            text()
        elif t == 9:
            element, count = take("I"), take("Q")
            if element in sizes:
                pos += sizes[element] * count
            else:
                for _ in range(count):
                    skip(element)
        else:
            pos += sizes[t]

    if head[:4] != b"GGUF" or struct.unpack_from("<I", head, 4)[0] != 3:
        raise ValueError(f"{path}: not a version 3 GGUF file")
    pos = 8
    tensor_count, metadata_count = take("Q"), take("Q")
    alignment = 32
    for _ in range(metadata_count):
        key, t = text(), take("I")
        if key == "general.alignment":
            alignment = struct.unpack_from("<I", head, pos)[0]
        skip(t)
    tensors = {}
    for _ in range(tensor_count):
        name = text()
        dims = [take("Q") for _ in range(take("I"))]
        tensors[name] = (dims, take("I"), take("Q"))
    start = (pos + alignment - 1) // alignment * alignment
    return {name: (dims, t, start + offset) for name, (dims, t, offset) in tensors.items()}


def fit_blocks(blocks):
    """Fits each row of 32 weights as step * q, q in -8..7. Returns f16
    steps and int8 q. Starts from the largest magnitude over 8 and over 7,
    alternates rounding with a least-squares step, and keeps the better."""
    x = blocks.astype(np.float64)
    amax = np.abs(x).max(axis=1, keepdims=True)
    amax[amax == 0] = 1.0
    best_step = best_q = None
    best_err = np.full((len(x), 1), np.inf)
    for k in (8.0, 7.0):
        step = amax / k
        for _ in range(4):
            q = np.clip(np.rint(x / step), -8, 8)
            qq = (q * q).sum(axis=1, keepdims=True)
            step = np.where(qq > 0, (x * q).sum(axis=1, keepdims=True) / np.maximum(qq, 1), step)
        err = ((q * step - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_step = np.where(better, step, step if best_step is None else best_step)
        best_q = np.where(better, q, q if best_q is None else best_q)
    # q4_0 holds -8..7; a block reaching +8 but not -8 fits with the step negated.
    flip = (best_q.max(axis=1, keepdims=True) > 7) & (best_q.min(axis=1, keepdims=True) > -8)
    best_step = np.where(flip, -best_step, best_step)
    step16 = best_step.astype(np.float16)
    wide = step16.astype(np.float64)
    wide[wide == 0] = 1.0
    q = np.clip(np.rint(x / wide), -8, 7).astype(np.int8)
    return step16.ravel(), q


def llama_blocks(blocks):
    """llama.cpp's q4_0 rule, for comparison: the largest-magnitude value over -8."""
    x = blocks.astype(np.float64)
    signed = np.take_along_axis(x, np.abs(x).argmax(axis=1)[:, None], axis=1)
    step16 = (signed / -8).astype(np.float16)
    wide = step16.astype(np.float64)
    wide[wide == 0] = 1.0
    return step16.ravel(), np.clip(np.rint(x / wide), -8, 7).astype(np.int8)


def pack(steps, q):
    """Encodes q4_0 blocks: an f16 step, then 16 bytes of nibbles holding
    q + 8, inputs 0-15 in the low nibbles and 16-31 in the high."""
    nibbles = (q + 8).astype(np.uint8)
    out = np.empty((len(q), 18), dtype=np.uint8)
    out[:, :2] = steps.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = nibbles[:, :16] | (nibbles[:, 16:] << 4)
    return out


def relative_error(blocks, steps, q):
    approx = q.astype(np.float64) * steps.astype(np.float64)[:, None]
    return np.linalg.norm(approx - blocks) / np.linalg.norm(blocks)


def matrices(weights, prefix):
    for name in sorted(weights.header):
        for hf in GGUF_TO_HF.values():
            if name.startswith(prefix) and name.endswith(hf + ".weight"):
                yield name


def check(path, limit):
    weights = SafeTensors(path)
    prefix = "model.language_model.layers."
    for name in list(matrices(weights, prefix))[:limit]:
        blocks = weights.get(name).reshape(-1, 32)
        ours = relative_error(blocks, *fit_blocks(blocks))
        theirs = relative_error(blocks, *llama_blocks(blocks))
        print(f"{name[len(prefix):]:40} fitted {ours:.4f}   llama.cpp rule {theirs:.4f}")


def refit(gguf_path, checkpoint_path, output_path):
    tensors = read_gguf_tensors(gguf_path)
    weights = SafeTensors(checkpoint_path)
    temporary = output_path.with_name(output_path.name + ".tmp")
    shutil.copyfile(gguf_path, temporary)
    out = np.memmap(temporary, dtype=np.uint8, mode="r+")
    worst, total, count = 0.0, 0.0, 0
    before_total = 0.0
    for name, (dims, kind, offset) in sorted(tensors.items()):
        if kind != GGML_Q4_0:
            continue
        _, layer, role, _ = name.split(".")
        hf = f"model.language_model.layers.{layer}.{GGUF_TO_HF[role]}.weight"
        w = weights.get(hf)
        if list(w.shape) != [dims[1], dims[0]]:
            raise ValueError(f"{name} is {dims}, but {hf} is {list(w.shape)}")
        blocks = w.reshape(-1, 32)
        size = len(blocks) * 18
        old = out[offset:offset + size].reshape(-1, 18)
        old_steps = old[:, :2].copy().view("<f2").ravel()
        old_nibbles = np.concatenate([old[:, 2:] & 15, old[:, 2:] >> 4], axis=1).astype(np.int8) - 8
        before = relative_error(blocks, old_steps, old_nibbles)
        if before > 0.2:
            raise ValueError(f"{name} does not match {hf} (relative error {before:.3f})")
        steps, q = fit_blocks(blocks)
        error = relative_error(blocks, steps, q)
        out[offset:offset + size] = pack(steps, q).ravel()
        worst, total, before_total, count = max(worst, error), total + error, before_total + before, count + 1
        print(f"{name:32} {before:.4f} -> {error:.4f}", file=sys.stderr)
    out.flush()
    del out
    os.replace(temporary, output_path)
    print(f"{count} q4_0 tensors: mean relative error {before_total / count:.4f} -> {total / count:.4f}, worst {worst:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", type=Path, nargs="?")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--out", type=Path)
    parser.add_argument("--check", type=int, metavar="N", help="report the fit for the first N matrices")
    args = parser.parse_args()
    if args.check:
        check(args.checkpoint, args.check)
    elif args.gguf and args.out:
        refit(args.gguf, args.checkpoint, args.out)
    else:
        parser.error("give a GGUF and -o, or --check")


if __name__ == "__main__":
    main()
