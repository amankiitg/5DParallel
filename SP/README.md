# Sequence Parallelism Tutorial - Companion Guide

## Overview

This tutorial demonstrates **Sequence Parallelism (SP)**, a technique that reduces activation memory when combined with Tensor Parallelism (TP).

## What's in the Tutorial

### Files

| File | Description |
|------|-------------|
| `sequence_parallelism_tutorial.ipynb` | Main notebook - run this |
| `config.py` | Model configuration |
| `vanilla_tp.py` | Vanilla TP implementation (baseline) |
| `tp_sp.py` | TP + SP implementation |
| `trace_shapes.py` | Visualizes tensor shapes |
| `analyze_comm.py` | Analyzes communication patterns |

### How to Run on RunPod

1. **Create a Pod** with 2+ GPUs (e.g., 2x RTX 4090 or 2x A100)
2. **Upload all files** to the same directory
3. **Open Jupyter Lab** and run the notebook

---

## The Core Idea

### Problem: Vanilla TP Wastes Memory

In vanilla Tensor Parallelism, after a row-parallel layer (like W_o or W2), we use ALL-REDUCE to combine partial results:

```
GPU-0: partial_0 ─┐
                  ├─ ALL-REDUCE ─> full result on BOTH GPUs
GPU-1: partial_1 ─┘
```

Then LayerNorm, Dropout, and Residual connections operate on the **full tensor** `(B, S, h)` on **every GPU**. This is redundant!

### Solution: Sequence Parallelism

Instead of ALL-REDUCE (which gives full tensor to everyone), use REDUCE-SCATTER (which gives each GPU 1/N of the result):

```
GPU-0: partial_0 ─┐
                  ├─ REDUCE-SCATTER ─> GPU-0 gets tokens 0 to S/2
GPU-1: partial_1 ─┘                    GPU-1 gets tokens S/2 to S
```

Now LayerNorm, Dropout, Residuals operate on `(B, S/N, h)` instead of `(B, S, h)`.

Before the next column-parallel layer, we use ALL-GATHER to reconstruct the full sequence:

```
GPU-0: tokens 0 to S/2   ─┐
                          ├─ ALL-GATHER ─> full (B, S, h) on both GPUs
GPU-1: tokens S/2 to S   ─┘
```

### Why This Works

ALL-REDUCE internally = REDUCE-SCATTER + ALL-GATHER

So we're doing the **same communication**, just **separated in time** to enable SP-region operations in between!

---

## Visual Comparison

### Vanilla TP Flow (per transformer block)

```
Input (B,S,h) ─> LayerNorm (B,S,h) ─> W_qkv (B,S,h/N) ─> Attention ─> W_o ─> ALL-REDUCE ─> (B,S,h)
     full            full             TP region         TP region         sum           full

                 ─> Dropout (B,S,h) ─> Residual (B,S,h) ─> ...
                         full              full
```

Memory for LayerNorm, Dropout, Residuals: `B * S * h` elements each (full)

### TP + SP Flow (per transformer block)

```
Input (B,S/N,h) ─> LayerNorm (B,S/N,h) ─> ALL-GATHER ─> (B,S,h) ─> W_qkv ─> Attention ─> W_o ─> REDUCE-SCATTER ─> (B,S/N,h)
     SP region         SP region           expand        full      TP region            compress        SP region

                   ─> Dropout (B,S/N,h) ─> Residual (B,S/N,h) ─> ...
                          SP region            SP region
```

Memory for LayerNorm, Dropout, Residuals: `B * S/N * h` elements each (1/N of vanilla!)

---

## Expected Results

With 2 GPUs (N=2), you should see:

### Memory

| Component | Vanilla TP | TP + SP | Savings |
|-----------|-----------|---------|---------|
| LayerNorm activations | (B, 512, 1024) | (B, 256, 1024) | 50% |
| Dropout activations | (B, 512, 1024) | (B, 256, 1024) | 50% |
| Residual tensors | (B, 512, 1024) | (B, 256, 1024) | 50% |
| **Peak memory** | Higher | **~30% lower** | Significant! |

### Throughput

Should be approximately the same (within noise), because:
- Same total computation
- Same total communication
- Small overhead from separate kernel launches

---

## When to Use Sequence Parallelism

**Always use SP when using TP.** The overhead is negligible and the memory savings are substantial.

SP is implemented in:
- Megatron-LM
- DeepSpeed
- NeMo
- Most serious distributed training frameworks

---

## What SP Doesn't Help

### 1. Attention Scores

The attention score matrix `(B, H/N, S, S)` is computed in the TP region and scales as O(S²). SP doesn't help here.

For very long sequences, you need **Context Parallelism** which shards the attention computation along the sequence dimension.

### 2. Model Too Large for TP=8

If your model doesn't fit with TP=8 (within a single node), you'd need TP>8 which crosses node boundaries and becomes very slow.

For this, you need **Pipeline Parallelism** which splits layers across nodes instead of splitting each layer.

---

## Troubleshooting

### "No module named config"

Make sure all .py files are in the same directory as the notebook.

### NCCL errors

- Make sure you have 2+ GPUs
- Check `nvidia-smi` to verify GPUs are visible
- Try reducing batch size if OOM

### Results look similar

With a small model on fast GPUs, the memory difference might be hard to see. The benefit is more pronounced with:
- Larger models (more layers)
- Longer sequences
- Larger batch sizes

---

## Further Reading

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Original TP implementation
- [Sequence Parallelism Paper](https://arxiv.org/abs/2105.13120) - Detailed analysis of SP
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) - SP + selective activation recomputation
