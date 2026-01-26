"""
Batch Prompt Enhancement for GenEval

Batch process prompts from evaluation_metadata.jsonl using PromptEnhancerV2.

Usage:
    python batch_enhance_prompts.py \
        --model /path/to/prompt_enhancer_model \
        --input ../prompts/evaluation_metadata.jsonl \
        --output ../prompts/enhanced_prompts.jsonl

Multi-GPU:
    for i in {0..3}; do
        CUDA_VISIBLE_DEVICES=$i python batch_enhance_prompts.py \
            --model /path/to/model \
            --input ../prompts/evaluation_metadata.jsonl \
            --output ../prompts/enhanced_prompts.jsonl \
            --gpu-id $i --num-gpus 4 &
    done
    wait
"""

import argparse
import json
import os
import sys
import time
import filelock
from pathlib import Path

# Add parent path to import PromptEnhancerV2
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "inference"))

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Batch enhance prompts for GenEval")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PromptEnhancer model (Qwen2.5-VL based)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../prompts/evaluation_metadata.jsonl",
        help="Input JSONL file with prompts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../prompts/enhanced_prompts.jsonl",
        help="Output JSONL file for enhanced prompts"
    )
    parser.add_argument(
        "--sys-prompt",
        type=str,
        default="Please rewrite the following prompt with more details for text-to-image generation:",
        help="System prompt for enhancement"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate"
    )
    # Multi-GPU arguments
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for this process"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Total number of GPUs"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for processing"
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for processing"
    )
    return parser.parse_args()


def load_input_prompts(input_file: str):
    """Load prompts from input JSONL file."""
    prompts = []
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data)
    return prompts


def load_existing_results(output_file: str) -> dict:
    """Load already processed results to support resuming."""
    results = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "index" in data:
                        results[data["index"]] = data
                except json.JSONDecodeError:
                    continue
    return results


def save_result(output_file: str, result: dict, lock_file: str):
    """Save a single result with file locking for multi-GPU safety."""
    lock = filelock.FileLock(lock_file, timeout=30)
    with lock:
        with open(output_file, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def get_indices_for_gpu(total: int, gpu_id: int, num_gpus: int, start: int = 0, end: int = None):
    """Get indices for this GPU (round-robin distribution)."""
    if end is None:
        end = total
    all_indices = list(range(start, end))
    return [idx for i, idx in enumerate(all_indices) if i % num_gpus == gpu_id]


def main():
    args = parse_args()

    # Import here to avoid loading model before args are parsed
    from prompt_enhancer_v2 import PromptEnhancerV2

    print(f"[GPU {args.gpu_id}] Loading model from: {args.model}")
    enhancer = PromptEnhancerV2(
        models_root_path=args.model,
        device_map="auto"
    )

    # Load input prompts
    input_file = args.input
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.path.dirname(__file__), input_file)

    prompts = load_input_prompts(input_file)
    print(f"[GPU {args.gpu_id}] Loaded {len(prompts)} prompts")

    # Setup output file
    output_file = args.output
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(__file__), output_file)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    lock_file = output_file + ".lock"

    # Load existing results for resuming
    existing = load_existing_results(output_file)
    print(f"[GPU {args.gpu_id}] Found {len(existing)} existing results")

    # Get indices for this GPU
    end_idx = args.end_index if args.end_index is not None else len(prompts)
    indices = get_indices_for_gpu(len(prompts), args.gpu_id, args.num_gpus, args.start_index, end_idx)

    # Filter out already processed
    indices = [i for i in indices if i not in existing]
    print(f"[GPU {args.gpu_id}/{args.num_gpus}] Processing {len(indices)} prompts")

    # Process prompts
    for idx in tqdm(indices, desc=f"GPU {args.gpu_id}", position=args.gpu_id):
        prompt_data = prompts[idx]
        original_prompt = prompt_data["prompt"]

        start_time = time.time()
        error_msg = None
        try:
            enhanced_prompt = enhancer.predict(
                prompt_cot=original_prompt,
                sys_prompt=args.sys_prompt,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"[GPU {args.gpu_id}] Error at index {idx}: {e}")
            enhanced_prompt = original_prompt  # Fallback to original
            error_msg = str(e)

        elapsed = time.time() - start_time

        # Build result
        result = {
            "index": idx,
            "tag": prompt_data.get("tag", ""),
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt,
            "time_seconds": round(elapsed, 2),
        }
        if error_msg:
            result["status"] = "error"
            result["error"] = error_msg
        else:
            result["status"] = "success"

        # Save immediately (supports resume)
        save_result(output_file, result, lock_file)

        if idx % 10 == 0:
            print(f"[GPU {args.gpu_id}] {idx}: '{original_prompt[:40]}...' -> '{enhanced_prompt[:40]}...' ({elapsed:.1f}s)")

    print(f"[GPU {args.gpu_id}] Done!")


if __name__ == "__main__":
    main()
