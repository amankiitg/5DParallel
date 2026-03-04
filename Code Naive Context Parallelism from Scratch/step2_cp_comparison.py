"""Step 2: Context Parallelism Comparison

Compares:
  1. Standard Attention: Full (S × S) attention matrix
  2. CP Attention: Smaller (S/CP × S) attention matrix per GPU

Memory savings come from each GPU only computing attention
for its local queries (S/CP) instead of all queries (S).
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
import time
import math
import json

def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    
    # Config
    B, H, D = 4, 16, 64  # batch, heads, head_dim
    
    if rank == 0:
        print('\n' + '=' * 65)
        print('  CONTEXT PARALLELISM COMPARISON')
        print('=' * 65)
        print(f'  GPUs: {ws}')
        print(f'  Batch: {B}, Heads: {H}, Head dim: {D}')
        print('=' * 65 + '\n')
    
    all_results = {}
    
    for S in [1024, 2048, 4096]:
        S_local = S // ws
        
        if rank == 0:
            print(f'--- Sequence Length: {S} ---')
        
        # ============================================================
        # TEST 1: Standard Attention (No CP)
        # Each GPU computes full (S × S) attention
        # ============================================================
        Q = torch.randn(B, H, S, D, device=device)
        K = torch.randn(B, H, S, D, device=device)
        V = torch.randn(B, H, S, D, device=device)
        
        # Warmup
        for _ in range(3):
            attn = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
            out = F.softmax(attn, dim=-1) @ V
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(10):
            attn = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
            out = F.softmax(attn, dim=-1) @ V
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        mem_no_cp = torch.cuda.max_memory_allocated() / 1024**2
        time_no_cp = (t1 - t0) / 10 * 1000
        
        del Q, K, V, attn, out
        torch.cuda.empty_cache()
        
        # ============================================================
        # TEST 2: CP-Style Attention
        # Each GPU computes (S/CP × S) attention for its local queries
        # ============================================================
        Q_local = torch.randn(B, H, S_local, D, device=device)
        K_full = torch.randn(B, H, S, D, device=device)
        V_full = torch.randn(B, H, S, D, device=device)
        
        # Warmup
        for _ in range(3):
            attn = (Q_local @ K_full.transpose(-2, -1)) / math.sqrt(D)
            out = F.softmax(attn, dim=-1) @ V_full
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(10):
            attn = (Q_local @ K_full.transpose(-2, -1)) / math.sqrt(D)
            out = F.softmax(attn, dim=-1) @ V_full
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        mem_cp = torch.cuda.max_memory_allocated() / 1024**2
        time_cp = (t1 - t0) / 10 * 1000
        
        del Q_local, K_full, V_full, attn, out
        torch.cuda.empty_cache()
        
        # Print results
        if rank == 0:
            reduction = (mem_no_cp - mem_cp) / mem_no_cp * 100
            print(f'  No CP:   attn=({S}×{S})       mem={mem_no_cp:7.1f} MB  time={time_no_cp:6.2f} ms')
            print(f'  CP={ws}:    attn=({S_local}×{S})     mem={mem_cp:7.1f} MB  time={time_cp:6.2f} ms')
            print(f'  Reduction: {reduction:.1f}%\n')
            
            all_results[str(S)] = {
                'no_cp': {'mem_mb': round(mem_no_cp, 1), 'time_ms': round(time_no_cp, 2)},
                'with_cp': {'mem_mb': round(mem_cp, 1), 'time_ms': round(time_cp, 2)},
                'reduction_pct': round(reduction, 1)
            }
    
    # Save results
    if rank == 0:
        with open('cp_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print('=' * 65)
        print('  SUMMARY')
        print('=' * 65)
        print(f"  {'Seq':<8} {'No CP Mem':>12} {'CP Mem':>12} {'Reduction':>12}")
        print('  ' + '-' * 44)
        for s, r in all_results.items():
            print(f"  {s:<8} {r['no_cp']['mem_mb']:>10.1f} MB {r['with_cp']['mem_mb']:>10.1f} MB {r['reduction_pct']:>10.1f}%")
        
        print('\n' + '=' * 65)
        print('  KEY INSIGHT')
        print('=' * 65)
        print(f'''
  Standard Attention:
    Each GPU: Q @ K.T  →  ({B}, {H}, S, S) attention matrix
    Memory: O(S²)

  Context Parallel Attention (CP={ws}):
    Each GPU: Q_local @ K_full.T  →  ({B}, {H}, S/{ws}, S) attention matrix
    Memory: O(S²/{ws})  →  {ws}x smaller!

  This is why CP is ESSENTIAL for long sequences (32K, 64K, 128K+).
  Without CP, attention memory explodes and you run out of GPU memory.
''')
        print('=' * 65)
        print('  Results saved to cp_results.json')
        print('  Run: python step3_plot.py')
        print('=' * 65 + '\n')
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
