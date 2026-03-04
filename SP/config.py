"""Shared configuration for Sequence Parallelism tutorial."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import time
import json

# Model Configuration
D_MODEL = 1024
N_HEADS = 16
D_HEAD = D_MODEL // N_HEADS  # 64
D_FF = 4096
N_LAYERS = 8
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 1024

# Benchmark Configuration
BATCH_SIZE = 4
SEQ_LEN = 512
NUM_WARMUP = 3
NUM_BENCHMARK = 10

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_memory_mb(device=0):
    return torch.cuda.memory_allocated(device) / 1024 / 1024

def get_peak_memory_mb(device=0):
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024

def reset_memory_stats(device=0):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
