"""
Qwen-Image Generation Script for GenEval Benchmark

Supports two modes:
1. Original prompts from evaluation_metadata.jsonl
2. Enhanced prompts from PromptEnhancer

Multi-GPU support:
  # Run on 4 GPUs in parallel
  for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python qwen_image_generate.py metadata.jsonl \\
      --gpu-id $i --num-gpus 4 --outdir outputs/qwen &
  done
  wait
"""

import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import QwenImagePipeline


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Qwen-Image for GenEval")
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--enhanced-prompts",
        type=str,
        default=None,
        help="JSONL file containing enhanced prompts from PromptEnhancer. "
             "Each line should have 'enhanced_prompt' field"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-Image",
        help="Huggingface model name for Qwen-Image"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/qwen_image",
        help="Directory to write results to"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=" ",
        help="Negative prompt for guidance (use empty space for CFG)"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=1024,
        help="Image height in pixels"
    )
    parser.add_argument(
        "--W",
        type=int,
        default=1024,
        help="Image width in pixels"
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="True CFG scale for guidance"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (increase if GPU memory allows)"
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip saving grid image"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for resuming generation"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for partial generation"
    )
    # Multi-GPU arguments
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID for this process (0-indexed)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Total number of GPUs for parallel generation"
    )
    return parser.parse_args()


def load_model(model_name: str, device: torch.device) -> QwenImagePipeline:
    """Load Qwen-Image pipeline."""
    print(f"Loading model: {model_name}")
    pipe = QwenImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(device)
    return pipe


def load_prompts(metadata_file: str, enhanced_file: str = None):
    """Load prompts from metadata and optionally enhanced prompts."""
    with open(metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    enhanced_prompts = None
    if enhanced_file and os.path.exists(enhanced_file):
        print(f"Loading enhanced prompts from: {enhanced_file}")
        with open(enhanced_file) as fp:
            enhanced_data = [json.loads(line) for line in fp]
        # Use "index" field from data as key (not enumerate line number)
        enhanced_prompts = {
            item["index"]: item.get("enhanced_prompt", item.get("prompt"))
            for item in enhanced_data
            if "index" in item
        }

    return metadatas, enhanced_prompts


def generate_images(
    pipe: QwenImagePipeline,
    prompt: str,
    n_samples: int,
    batch_size: int,
    args,
    generator: torch.Generator
) -> list:
    """Generate images for a single prompt."""
    all_images = []
    sample_count = 0

    while sample_count < n_samples:
        current_batch = min(batch_size, n_samples - sample_count)

        images = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.H,
            width=args.W,
            num_inference_steps=args.steps,
            true_cfg_scale=args.true_cfg_scale,
            num_images_per_prompt=current_batch,
            generator=generator,
        ).images

        all_images.extend(images)
        sample_count += len(images)

    return all_images


def save_outputs(
    images: list,
    metadata: dict,
    outpath: str,
    prompt_used: str,
    skip_grid: bool = False
):
    """Save generated images and metadata."""
    os.makedirs(outpath, exist_ok=True)
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # Save metadata with the prompt actually used
    output_metadata = metadata.copy()
    output_metadata["prompt_used"] = prompt_used
    with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
        json.dump(output_metadata, fp)

    # Save individual images
    for i, img in enumerate(images):
        img.save(os.path.join(sample_path, f"{i:05d}.png"))

    # Save grid
    if not skip_grid and len(images) > 0:
        tensors = torch.stack([ToTensor()(img) for img in images], 0)
        nrow = int(np.ceil(np.sqrt(len(images))))
        grid = make_grid(tensors, nrow=nrow)
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        grid.save(os.path.join(outpath, "grid.png"))


def get_indices_for_gpu(total_count: int, gpu_id: int, num_gpus: int, start_idx: int = 0, end_idx: int = None):
    """Split indices across GPUs for parallel processing."""
    if end_idx is None:
        end_idx = total_count
    all_indices = list(range(start_idx, end_idx))
    # Round-robin distribution
    return [idx for i, idx in enumerate(all_indices) if i % num_gpus == gpu_id]


def main():
    args = parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[GPU {args.gpu_id}] Using device: {device}")

    # Load model
    pipe = load_model(args.model, device)

    # Load prompts
    metadatas, enhanced_prompts = load_prompts(args.metadata_file, args.enhanced_prompts)

    # Determine range with multi-GPU support
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(metadatas)

    if args.num_gpus > 1:
        indices = get_indices_for_gpu(len(metadatas), args.gpu_id, args.num_gpus, start_idx, end_idx)
        print(f"[GPU {args.gpu_id}/{args.num_gpus}] Processing {len(indices)} prompts")
    else:
        indices = list(range(start_idx, end_idx))

    print(f"[GPU {args.gpu_id}] Output directory: {args.outdir}")
    print(f"[GPU {args.gpu_id}] Using enhanced prompts: {args.enhanced_prompts is not None}")

    for index in tqdm(indices, desc=f"GPU {args.gpu_id}", position=args.gpu_id):
        metadata = metadatas[index]
        # Use index-based seed to ensure different prompts have different random states
        prompt_seed = args.seed + index
        seed_everything(prompt_seed)
        generator = torch.Generator(device=device).manual_seed(prompt_seed)

        # Determine which prompt to use
        if enhanced_prompts and index in enhanced_prompts:
            prompt = enhanced_prompts[index]
        else:
            prompt = metadata["prompt"]

        outpath = os.path.join(args.outdir, f"{index:05d}")

        # Skip if already generated
        if os.path.exists(os.path.join(outpath, "metadata.jsonl")):
            continue

        print(f"[GPU {args.gpu_id}] Prompt ({index:>3}/{len(metadatas)}): '{prompt[:60]}...'")

        # Generate images
        images = generate_images(
            pipe=pipe,
            prompt=prompt,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            args=args,
            generator=generator
        )

        # Save outputs
        save_outputs(
            images=images,
            metadata=metadata,
            outpath=outpath,
            prompt_used=prompt,
            skip_grid=args.skip_grid
        )

    print(f"[GPU {args.gpu_id}] Done.")


if __name__ == "__main__":
    main()
