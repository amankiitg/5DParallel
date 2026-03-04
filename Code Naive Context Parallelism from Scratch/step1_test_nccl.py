"""Step 1: Verify NCCL works on fresh pod"""
import torch
import torch.distributed as dist

dist.init_process_group('nccl')
rank = dist.get_rank()
ws = dist.get_world_size()
torch.cuda.set_device(rank)

print(f'[GPU {rank}] Initialized', flush=True)

x = torch.tensor([rank + 1.0], device=f'cuda:{rank}')
dist.all_reduce(x)

print(f'[GPU {rank}] all_reduce result: {x.item()} (expected: {ws * (ws + 1) / 2})', flush=True)

if rank == 0:
    print('\n✓ NCCL is working! Proceed to step 2.\n', flush=True)

dist.destroy_process_group()
