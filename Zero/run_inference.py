#!/usr/bin/env python3
"""
================================================================================
Inference — Test the Fine-tuned Model
================================================================================
Loads a model saved by deepspeed_train.py and generates responses to
instruction prompts. Shows before (base model) vs after (fine-tuned) comparison.

Usage:
    python run_inference.py --model_path ./output/zero2 \
                            --base_model EleutherAI/pythia-6.9b-deduped
================================================================================
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

TEST_PROMPTS = [
    "Explain the concept of data parallelism in distributed machine learning.",
    "Write a Python function that computes the Fibonacci sequence up to n terms.",
    "What are the key differences between TCP and UDP protocols?",
    "Summarize the main causes of the French Revolution in three sentences.",
    "Write a haiku about machine learning.",
]


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Only decode the NEW tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def run_comparison(base_model_name, finetuned_path, prompts):
    """Compare base vs fine-tuned model responses."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load base model ───────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  Loading base model: {base_model_name}")
    print(f"{'═'*72}\n")

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()

    # ── Load fine-tuned model ─────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  Loading fine-tuned model: {finetuned_path}")
    print(f"{'═'*72}\n")

    ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token

    ft_model = AutoModelForCausalLM.from_pretrained(
        finetuned_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ft_model.eval()

    # ── Generate and compare ──────────────────────────────────────────
    for i, instruction in enumerate(prompts, 1):
        prompt = ALPACA_PROMPT.format(instruction=instruction)

        print(f"\n{'─'*72}")
        print(f"  PROMPT {i}: {instruction}")
        print(f"{'─'*72}")

        # Base model response
        print(f"\n  📌 BASE MODEL:")
        base_response = generate(base_model, base_tokenizer, prompt)
        for line in base_response.split('\n')[:10]:
            print(f"     {line}")
        if len(base_response.split('\n')) > 10:
            print(f"     ... (truncated)")

        # Fine-tuned response
        print(f"\n  ✅ FINE-TUNED MODEL:")
        ft_response = generate(ft_model, ft_tokenizer, prompt)
        for line in ft_response.split('\n')[:10]:
            print(f"     {line}")
        if len(ft_response.split('\n')) > 10:
            print(f"     ... (truncated)")

    # ── Cleanup ───────────────────────────────────────────────────────
    del base_model, ft_model
    torch.cuda.empty_cache()

    print(f"\n{'═'*72}")
    print(f"  Inference complete!")
    print(f"{'═'*72}\n")


def run_single(model_path, prompts):
    """Run inference on a single model (no comparison)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'═'*72}")
    print(f"  Loading model: {model_path}")
    print(f"{'═'*72}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    for i, instruction in enumerate(prompts, 1):
        prompt = ALPACA_PROMPT.format(instruction=instruction)

        print(f"\n{'─'*72}")
        print(f"  PROMPT {i}: {instruction}")
        print(f"{'─'*72}\n")

        response = generate(model, tokenizer, prompt)
        for line in response.split('\n'):
            print(f"  {line}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model for comparison (optional)")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="Custom prompts (default: built-in test set)")
    args = parser.parse_args()

    prompts = args.prompts or TEST_PROMPTS

    if args.base_model:
        run_comparison(args.base_model, args.model_path, prompts)
    else:
        run_single(args.model_path, prompts)


if __name__ == "__main__":
    main()
