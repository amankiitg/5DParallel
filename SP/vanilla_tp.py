"""Vanilla Tensor Parallelism - Baseline for comparison.

In vanilla TP:
- Column-parallel layers split output dimension, no communication
- Row-parallel layers split input dimension, ALL-REDUCE to combine
- LayerNorm, Dropout, Residuals operate on FULL (B, S, h) tensors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import time
import json
import sys
sys.path.insert(0, ".")
from config import *


# ============================================================================
#  AUTOGRAD FUNCTIONS FOR VANILLA TP
# ============================================================================

class _CopyToTPRegion(torch.autograd.Function):
    """Identity forward, all-reduce backward."""
    @staticmethod
    def forward(ctx, x):
        return x
    
    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


class _ReduceFromTPRegion(torch.autograd.Function):
    """All-reduce forward, identity backward."""
    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x
    
    @staticmethod
    def backward(ctx, grad):
        return grad


# ============================================================================
#  LINEAR LAYERS
# ============================================================================

class ColumnParallelLinear(nn.Module):
    """Split weight columns (output dim) across GPUs. No comm in forward."""
    
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.world_size = dist.get_world_size()
        assert d_out % self.world_size == 0
        self.d_out_local = d_out // self.world_size
        
        self.weight = nn.Parameter(torch.empty(self.d_out_local, d_in))
        self.bias = nn.Parameter(torch.empty(self.d_out_local)) if bias else None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.shape[1])
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        x = _CopyToTPRegion.apply(x)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Split weight rows (input dim) across GPUs. ALL-REDUCE in forward."""
    
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert d_in % self.world_size == 0
        self.d_in_local = d_in // self.world_size
        
        self.weight = nn.Parameter(torch.empty(d_out, self.d_in_local))
        self.bias = None
        if bias and self.rank == 0:
            self.bias = nn.Parameter(torch.empty(d_out))
            nn.init.uniform_(self.bias, -1/math.sqrt(d_in), 1/math.sqrt(d_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return _ReduceFromTPRegion.apply(out)  # ALL-REDUCE here!


# ============================================================================
#  ATTENTION
# ============================================================================

class VanillaTPAttention(nn.Module):
    """Multi-head attention with vanilla TP. ALL-REDUCE after W_o."""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.n_heads_local = n_heads // self.world_size
        self.d_head = d_model // n_heads
        self.d_local = self.n_heads_local * self.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        self.W_q = ColumnParallelLinear(d_model, d_model, bias=False)
        self.W_k = ColumnParallelLinear(d_model, d_model, bias=False)
        self.W_v = ColumnParallelLinear(d_model, d_model, bias=False)
        self.W_o = RowParallelLinear(d_model, d_model, bias=False)
    
    def forward(self, x):
        B, S, _ = x.shape
        
        Q = self.W_q(x).view(B, S, self.n_heads_local, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads_local, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads_local, self.d_head).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, self.d_local)
        return self.W_o(out)  # ALL-REDUCE happens here


# ============================================================================
#  FFN
# ============================================================================

class VanillaTPFFN(nn.Module):
    """Feed-forward with vanilla TP. ALL-REDUCE after W2."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = ColumnParallelLinear(d_model, d_ff)
        self.W2 = RowParallelLinear(d_ff, d_model)
    
    def forward(self, x):
        return self.W2(F.gelu(self.W1(x)))


# ============================================================================
#  TRANSFORMER BLOCK
# ============================================================================

class VanillaTPBlock(nn.Module):
    """Transformer block with vanilla TP.
    
    LayerNorm, Dropout, Residuals all operate on FULL (B, S, h) tensors.
    This is where memory is "wasted" compared to TP+SP.
    """
    
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # FULL (B, S, h)
        self.attn = VanillaTPAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)  # FULL (B, S, h)
        self.ffn = VanillaTPFFN(d_model, d_ff)
        self.dropout = nn.Dropout(0.1)    # FULL (B, S, h)
    
    def forward(self, x):
        # x: (B, S, h) - FULL tensor, same on all GPUs
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x  # (B, S, h) - FULL


# ============================================================================
#  FULL MODEL
# ============================================================================

class VanillaTPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            VanillaTPBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = ColumnParallelLinear(D_MODEL, VOCAB_SIZE, bias=False)
    
    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


class _AllGatherForLogits(torch.autograd.Function):
    """All-gather that preserves gradients for the local chunk."""
    @staticmethod
    def forward(ctx, local_logits):
        ws = dist.get_world_size()
        ctx.ws = ws
        ctx.rank = dist.get_rank()
        gathered = [torch.zeros_like(local_logits) for _ in range(ws)]
        dist.all_gather(gathered, local_logits.contiguous())
        return torch.cat(gathered, dim=-1)
    
    @staticmethod
    def backward(ctx, grad_full):
        # Each rank only needs gradient for its own chunk
        chunk_size = grad_full.shape[-1] // ctx.ws
        start = ctx.rank * chunk_size
        end = start + chunk_size
        return grad_full[..., start:end].contiguous()


def tp_cross_entropy(logits_local, labels, vocab_size):
    """Cross-entropy when vocab is split across GPUs."""
    full_logits = _AllGatherForLogits.apply(logits_local)
    return F.cross_entropy(full_logits.view(-1, vocab_size), labels.view(-1))


# ============================================================================
#  BENCHMARK
# ============================================================================

def main():
    import datetime
    # Try NCCL first, fall back to GLOO if it fails
    try:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=60))
    except Exception as e:
        print(f"NCCL failed ({e}), trying GLOO...")
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    reset_memory_stats(device)
    torch.manual_seed(42)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print(f"  VANILLA TENSOR PARALLELISM (TP={world_size})")
        print("=" * 70)
    
    model = VanillaTPModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    mem_model = get_memory_mb(device)
    n_params = count_parameters(model)
    
    torch.manual_seed(123)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    
    # Warmup
    for _ in range(NUM_WARMUP):
        logits = model(input_ids)
        loss = tp_cross_entropy(logits, labels, VOCAB_SIZE)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    reset_memory_stats(device)
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark
    fwd_times, bwd_times, step_times = [], [], []
    for _ in range(NUM_BENCHMARK):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        
        logits = model(input_ids)
        loss = tp_cross_entropy(logits, labels, VOCAB_SIZE)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        step_times.append(t3 - t0)
    
    peak_mem = get_peak_memory_mb(device)
    
    # Report from all ranks
    for r in range(world_size):
        if rank == r:
            print(f"  [GPU {rank}] params: {n_params:,} | model: {mem_model:.1f} MB | peak: {peak_mem:.1f} MB")
        dist.barrier()
    
    if rank == 0:
        avg = lambda lst: sum(lst) / len(lst)
        results = {
            "mode": "vanilla_tp",
            "world_size": world_size,
            "params_per_gpu": n_params,
            "mem_model_mb": round(mem_model, 2),
            "mem_peak_mb": round(peak_mem, 2),
            "fwd_ms": round(avg(fwd_times) * 1000, 2),
            "bwd_ms": round(avg(bwd_times) * 1000, 2),
            "step_ms": round(avg(step_times) * 1000, 2),
            "throughput": round(BATCH_SIZE * SEQ_LEN / avg(step_times), 1),
            "loss": round(loss.item(), 4),
        }
        
        with open("results_vanilla_tp.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  Forward:     {results['fwd_ms']:>8.2f} ms")
        print(f"  Backward:    {results['bwd_ms']:>8.2f} ms")
        print(f"  Full step:   {results['step_ms']:>8.2f} ms")
        print(f"  Throughput:  {results['throughput']:>8.0f} tok/s")
        print(f"  Peak memory: {results['mem_peak_mb']:>8.1f} MB")
        print("=" * 70)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
