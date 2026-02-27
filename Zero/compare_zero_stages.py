#!/usr/bin/env python3
"""Compare results across all ZeRO stages."""

import json
import glob
import os
import sys


def load_results(output_dir="./output"):
    results = []
    for fpath in sorted(glob.glob(os.path.join(output_dir, "results_zero*.json"))):
        with open(fpath) as f:
            results.append(json.load(f))
    return results


def print_comparison(results):
    if not results:
        print("No results found! Run the experiments first.")
        return

    print(f"\n{'═'*90}")
    print(f"  DeepSpeed ZeRO Stage Comparison — {results[0].get('model', 'Unknown')}")
    print(f"{'═'*90}\n")

    # Header
    print(f"  {'Stage':<10s} {'Peak Mem':>10s} {'Step Time':>12s} {'Throughput':>14s} "
          f"{'Train Loss':>12s} {'Eval Loss':>11s} {'Eval PPL':>10s}")
    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*14} {'─'*12} {'─'*11} {'─'*10}")

    for r in sorted(results, key=lambda x: x.get("zero_stage", 0)):
        stage = f"ZeRO-{r.get('zero_stage', '?')}"
        mem = r.get("peak_memory_gb", 0)
        step_ms = r.get("avg_step_time_ms", 0)
        tput = r.get("avg_throughput_tok_s", 0)
        train_loss = r.get("final_train_loss", 0)
        eval_loss = r.get("final_eval_loss", 0)
        eval_ppl = r.get("final_eval_ppl", 0)

        print(f"  {stage:<10s} {mem:>8.2f}GB {step_ms:>10.1f}ms "
              f"{tput:>12.0f} t/s {train_loss:>12.4f} {eval_loss:>11.4f} {eval_ppl:>10.2f}")

    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*14} {'─'*12} {'─'*11} {'─'*10}")

    # Analysis
    if len(results) >= 2:
        z0 = next((r for r in results if r.get("zero_stage") == 0), None)
        z3 = next((r for r in results if r.get("zero_stage") == 3), None)

        if z0 and z3:
            mem_saving = (1 - z3["peak_memory_gb"] / z0["peak_memory_gb"]) * 100
            speed_diff = (z3["avg_throughput_tok_s"] / z0["avg_throughput_tok_s"] - 1) * 100
            print(f"\n  KEY FINDINGS:")
            print(f"  • ZeRO-3 uses {abs(mem_saving):.1f}% {'less' if mem_saving > 0 else 'more'} "
                  f"peak memory than ZeRO-0")
            print(f"  • ZeRO-3 is {abs(speed_diff):.1f}% {'faster' if speed_diff > 0 else 'slower'} "
                  f"than ZeRO-0")
            print(f"  • All stages converge to similar loss (same math!)")

    print()

    # Try to create charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        results_sorted = sorted(results, key=lambda x: x.get("zero_stage", 0))
        stages = [f"ZeRO-{r['zero_stage']}" for r in results_sorted]
        memories = [r["peak_memory_gb"] for r in results_sorted]
        throughputs = [r["avg_throughput_tok_s"] for r in results_sorted]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Memory
        colors_mem = ["#ef4444" if m == max(memories) else "#22c55e" if m == min(memories) else "#4a9eed" for m in memories]
        axes[0].bar(stages, memories, color=colors_mem, edgecolor="white", linewidth=1.5)
        axes[0].set_ylabel("Peak Memory (GB)", fontsize=12)
        axes[0].set_title("GPU Memory Usage", fontsize=14, fontweight="bold")
        for i, v in enumerate(memories):
            axes[0].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")

        # Throughput
        colors_tput = ["#ef4444" if t == min(throughputs) else "#22c55e" if t == max(throughputs) else "#4a9eed" for t in throughputs]
        axes[1].bar(stages, throughputs, color=colors_tput, edgecolor="white", linewidth=1.5)
        axes[1].set_ylabel("Tokens/sec", fontsize=12)
        axes[1].set_title("Training Throughput", fontsize=14, fontweight="bold")
        for i, v in enumerate(throughputs):
            axes[1].text(i, v + 100, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")

        # Loss curves
        for r in results_sorted:
            stage = f"ZeRO-{r['zero_stage']}"
            loss_hist = r.get("loss_history", [])
            if loss_hist:
                axes[2].plot(range(1, len(loss_hist)+1), loss_hist, label=stage, linewidth=2)
        axes[2].set_xlabel("Step", fontsize=12)
        axes[2].set_ylabel("Training Loss", fontsize=12)
        axes[2].set_title("Loss Convergence (should overlap!)", fontsize=14, fontweight="bold")
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f"DeepSpeed ZeRO Comparison — {results[0].get('model', '')}", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        chart_path = os.path.join("./output", "zero_comparison.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart saved to {chart_path}")

    except ImportError:
        print("  (Install matplotlib for charts: pip install matplotlib)")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    results = load_results(output_dir)
    print_comparison(results)
