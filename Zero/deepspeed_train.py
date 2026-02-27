#!/usr/bin/env python3
"""
================================================================================
DeepSpeed ZeRO — Production-Level Training Script
================================================================================
Fine-tunes a 7B-class language model on instruction-following data using
DeepSpeed ZeRO. Supports stages 0–3 via config swap.

Features:
  ✓ BF16 mixed precision (optimized for B200/H100/A100)
  ✓ Gradient accumulation
  ✓ Cosine LR with linear warmup
  ✓ Periodic evaluation on held-out set
  ✓ Per-step memory & throughput tracking
  ✓ Proper model saving (handles ZeRO-3 parameter consolidation)
  ✓ Reproducible (seeded)
  ✓ Clean JSON metrics output for comparison

Usage:
  deepspeed --num_gpus=2 deepspeed_train.py \
      --deepspeed_config ds_zero2.json \
      --model_name EleutherAI/pythia-6.9b-deduped \
      --max_steps 500 --batch_size 4 --grad_accum 2

Author: Lecture materials for Data Parallelism course
================================================================================
"""

import os
import sys
import time
import json
import math
import argparse
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import deepspeed


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)

ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with further "
    "context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)


def parse_args():
    p = argparse.ArgumentParser(description="DeepSpeed ZeRO Training")
    # Model
    p.add_argument("--model_name", type=str,
                   default="EleutherAI/pythia-6.9b-deduped",
                   help="HuggingFace model name or path")
    # Training
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Micro batch size per GPU")
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Gradient accumulation steps")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=20,
                   help="Number of eval batches")
    # Output
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--save_model", action="store_true", default=True)
    p.add_argument("--run_name", type=str, default=None,
                   help="Name for this run (auto-detected from config if not set)")
    # DeepSpeed
    p.add_argument("--local_rank", type=int, default=-1)
    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


def set_seed(seed):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DATASET                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

class AlpacaDataset(Dataset):
    """
    Stanford Alpaca instruction-following dataset.
    Each example is formatted as a prompt + completion and tokenized
    to a fixed length with proper labels (loss only on response tokens).
    """

    def __init__(self, tokenizer, seq_len, split="train", max_samples=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Load Alpaca dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

        # Split: 95% train, 5% eval
        dataset = dataset.train_test_split(test_size=0.05, seed=42)
        raw = dataset["train"] if split == "train" else dataset["test"]

        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.examples = []
        for item in raw:
            # Format the prompt
            instruction = item["instruction"]
            inp = item.get("input", "")
            output = item["output"]

            if inp and inp.strip():
                text = ALPACA_PROMPT_WITH_INPUT.format(
                    instruction=instruction, input=inp, response=output
                )
            else:
                text = ALPACA_PROMPT.format(
                    instruction=instruction, response=output
                )

            # Find where the response starts (for label masking)
            prompt_part = text.split("### Response:\n")[0] + "### Response:\n"

            self.examples.append({
                "text": text,
                "prompt_len": len(tokenizer.encode(prompt_part)),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # Tokenize
        encoded = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels: mask the prompt portion (set to -100)
        # Only compute loss on the response tokens
        labels = input_ids.clone()
        prompt_len = min(example["prompt_len"], self.seq_len)
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Stack batch items into tensors."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def evaluate(model_engine, eval_dataloader, device, eval_steps):
    """Run evaluation and return average loss + perplexity."""
    model_engine.eval()
    total_loss = 0.0
    total_steps = 0

    eval_iter = iter(eval_dataloader)
    for _ in range(eval_steps):
        try:
            batch = next(eval_iter)
        except StopIteration:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        total_loss += outputs.loss.float().item()
        total_steps += 1

    model_engine.train()

    if total_steps == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_steps
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    return avg_loss, perplexity


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  COSINE LR SCHEDULER                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

class CosineWarmupScheduler:
    """Linear warmup → cosine decay to 10% of peak LR."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MAIN TRAINING LOOP                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Distributed init ──────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # ── Detect ZeRO stage from config ─────────────────────────────────
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
    run_name = args.run_name or f"zero{zero_stage}"

    if rank == 0:
        print(f"\n{'═'*72}")
        print(f"  DeepSpeed ZeRO Stage {zero_stage} — {args.model_name}")
        print(f"{'═'*72}")
        print(f"  World size:         {world_size} GPUs")
        print(f"  Micro batch/GPU:    {args.batch_size}")
        print(f"  Gradient accum:     {args.grad_accum}")
        print(f"  Effective batch:    {args.batch_size * world_size * args.grad_accum}")
        print(f"  Sequence length:    {args.seq_len}")
        print(f"  Max steps:          {args.max_steps}")
        print(f"  Learning rate:      {args.lr}")
        print(f"  Warmup steps:       {args.warmup_steps}")
        print(f"  Output dir:         {args.output_dir}/{run_name}")
        print(f"{'═'*72}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ─────────────────────────────────────────────────────────
    if rank == 0:
        print("  Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager",
    )

    # Enable gradient checkpointing to save activation memory
    model.gradient_checkpointing_enable()

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"  Model loaded: {num_params/1e9:.2f}B parameters")
        param_mem = num_params * 2 / (1024**3)  # bf16
        optim_mem = num_params * 12 / (1024**3)  # Adam: fp32 master + m + v
        grad_mem = num_params * 2 / (1024**3)    # bf16 grads
        total_mem = param_mem + optim_mem + grad_mem
        print(f"\n  ┌─ Memory Budget (before sharding) ──────────────┐")
        print(f"  │  Parameters (bf16):     {param_mem:>8.2f} GB              │")
        print(f"  │  Gradients (bf16):      {grad_mem:>8.2f} GB              │")
        print(f"  │  Optimizer (fp32 m+v):  {optim_mem:>8.2f} GB              │")
        print(f"  │  ─────────────────────────────────────────────  │")
        print(f"  │  Total per GPU (ZeRO-0): {total_mem:>7.2f} GB             │")
        print(f"  │  Total per GPU (ZeRO-1): {param_mem+grad_mem+optim_mem/world_size:>7.2f} GB             │")
        print(f"  │  Total per GPU (ZeRO-2): {param_mem+(grad_mem+optim_mem)/world_size:>7.2f} GB             │")
        print(f"  │  Total per GPU (ZeRO-3): {(param_mem+grad_mem+optim_mem)/world_size:>7.2f} GB             │")
        print(f"  └────────────────────────────────────────────────┘\n")

    # ── Dataset ───────────────────────────────────────────────────────
    if rank == 0:
        print("  Loading dataset...")

    train_dataset = AlpacaDataset(tokenizer, args.seq_len, split="train")
    eval_dataset = AlpacaDataset(tokenizer, args.seq_len, split="eval")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    eval_sampler = DistributedSampler(
        eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, sampler=eval_sampler,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True,
    )

    if rank == 0:
        print(f"  Train: {len(train_dataset)} examples")
        print(f"  Eval:  {len(eval_dataset)} examples\n")

    # ── DeepSpeed Init ────────────────────────────────────────────────
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config,
    )

    # LR scheduler (DeepSpeed manages the optimizer, we manage LR)
    scheduler = CosineWarmupScheduler(
        optimizer, args.warmup_steps, args.max_steps
    )

    # ── Metrics tracking ──────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    step_times = []
    losses = []
    eval_results = []
    memory_snapshots = []

    # ── Training ──────────────────────────────────────────────────────
    model_engine.train()
    data_iter = iter(train_dataloader)
    epoch = 0
    global_step = 0
    accum_loss = 0.0

    if rank == 0:
        print(f"  {'Step':>6s} | {'Loss':>8s} | {'LR':>10s} | {'Time/step':>10s} | "
              f"{'Tok/s':>10s} | {'GPU Mem':>8s} | {'Peak Mem':>8s}")
        print(f"  {'─'*6} | {'─'*8} | {'─'*10} | {'─'*10} | "
              f"{'─'*10} | {'─'*8} | {'─'*8}")

    torch.cuda.synchronize()
    t_start_total = time.perf_counter()
    step_t0 = time.perf_counter()

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            train_sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward + backward (DeepSpeed handles grad accum internally)
        outputs = model_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

        accum_loss += loss.float().item()

        # DeepSpeed tracks micro steps; we track optimizer steps
        if model_engine.is_gradient_accumulation_boundary():
            global_step += 1
            scheduler.step()

            torch.cuda.synchronize()
            step_t1 = time.perf_counter()
            step_time = step_t1 - step_t0

            avg_loss = accum_loss / args.grad_accum
            step_times.append(step_time)
            losses.append(avg_loss)

            # Memory snapshot
            current_mem = torch.cuda.memory_allocated(device) / (1024**3)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
            memory_snapshots.append({
                "step": global_step,
                "current_gb": round(current_mem, 2),
                "peak_gb": round(peak_mem, 2),
            })

            # Logging
            if rank == 0 and global_step % args.log_interval == 0:
                tokens_per_step = (
                    args.batch_size * args.seq_len * world_size * args.grad_accum
                )
                throughput = tokens_per_step / step_time
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  {global_step:6d} | {avg_loss:8.4f} | {lr:10.2e} | "
                    f"{step_time*1000:8.1f}ms | {throughput:8.0f} | "
                    f"{current_mem:6.2f}GB | {peak_mem:6.2f}GB"
                )

            # Evaluation
            if global_step % args.eval_interval == 0:
                eval_loss, eval_ppl = evaluate(
                    model_engine, eval_dataloader, device, args.eval_steps
                )
                eval_results.append({
                    "step": global_step,
                    "eval_loss": round(eval_loss, 4),
                    "eval_ppl": round(eval_ppl, 2),
                })
                if rank == 0:
                    print(
                        f"  {'':>6s}   ├── Eval Loss: {eval_loss:.4f} | "
                        f"Eval PPL: {eval_ppl:.2f}"
                    )

            accum_loss = 0.0
            step_t0 = time.perf_counter()

    torch.cuda.synchronize()
    t_end_total = time.perf_counter()

    # ── Final Evaluation ──────────────────────────────────────────────
    final_eval_loss, final_eval_ppl = evaluate(
        model_engine, eval_dataloader, device, args.eval_steps * 2
    )

    # ── Save Model ────────────────────────────────────────────────────
    output_path = os.path.join(args.output_dir, run_name)
    os.makedirs(output_path, exist_ok=True)

    if args.save_model:
        if rank == 0:
            print(f"\n  Saving model to {output_path}...")

        if zero_stage == 3:
            # ZeRO-3: params are distributed — need to consolidate
            model_engine.save_16bit_model(output_path, save_filename="model.safetensors")
        else:
            # ZeRO 0/1/2: params are local, save from rank 0
            if rank == 0:
                # Unwrap the model and save in HF format
                unwrapped = model_engine.module
                unwrapped.save_pretrained(
                    output_path,
                    safe_serialization=True,
                )

        # Save tokenizer from rank 0
        if rank == 0:
            tokenizer.save_pretrained(output_path)
            print(f"  ✓ Model saved to {output_path}")

    # ── Compile Results ───────────────────────────────────────────────
    peak_mem_final = torch.cuda.max_memory_allocated(device) / (1024**3)
    avg_step_time = sum(step_times[10:]) / max(1, len(step_times[10:]))  # skip warmup
    tokens_per_step = args.batch_size * args.seq_len * world_size * args.grad_accum
    avg_throughput = tokens_per_step / avg_step_time

    if rank == 0:
        results = {
            "run_name": run_name,
            "framework": f"DeepSpeed ZeRO Stage {zero_stage}",
            "model": args.model_name,
            "num_params_B": round(num_params / 1e9, 2),
            "world_size": world_size,
            "micro_batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * world_size * args.grad_accum,
            "seq_len": args.seq_len,
            "max_steps": args.max_steps,
            "learning_rate": args.lr,
            "zero_stage": zero_stage,
            "peak_memory_gb": round(peak_mem_final, 2),
            "avg_step_time_ms": round(avg_step_time * 1000, 1),
            "avg_throughput_tok_s": round(avg_throughput),
            "final_train_loss": round(losses[-1], 4) if losses else None,
            "final_eval_loss": round(final_eval_loss, 4),
            "final_eval_ppl": round(final_eval_ppl, 2),
            "total_time_s": round(t_end_total - t_start_total, 1),
            "loss_history": [round(l, 4) for l in losses],
            "eval_history": eval_results,
            "memory_history": memory_snapshots[:10] + memory_snapshots[-5:],
        }

        results_file = os.path.join(args.output_dir, f"results_{run_name}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'═'*72}")
        print(f"  RESULTS — DeepSpeed ZeRO Stage {zero_stage}")
        print(f"{'═'*72}")
        print(f"  Model:              {args.model_name} ({num_params/1e9:.2f}B)")
        print(f"  Peak GPU Memory:    {peak_mem_final:.2f} GB")
        print(f"  Avg Step Time:      {avg_step_time*1000:.1f} ms")
        print(f"  Avg Throughput:     {avg_throughput:.0f} tokens/sec")
        print(f"  Final Train Loss:   {losses[-1]:.4f}" if losses else "  N/A")
        print(f"  Final Eval Loss:    {final_eval_loss:.4f}")
        print(f"  Final Eval PPL:     {final_eval_ppl:.2f}")
        print(f"  Total Time:         {t_end_total - t_start_total:.1f}s")
        print(f"  Results saved to:   {results_file}")
        print(f"{'═'*72}\n")

    dist.barrier()
    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    main()
