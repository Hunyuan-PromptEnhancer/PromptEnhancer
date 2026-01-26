"""
Hunyuan-Image 2.1 Generation Script for GenEval Benchmark

Supports two modes:
1. Original prompts from evaluation_metadata.jsonl
2. Enhanced prompts from PromptEnhancer

Hunyuan-Image 2.1 uses Adaptive Projected Guidance (APG) + CFG.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import HunyuanImagePipeline


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using Hunyuan-Image 2.1 for GenEval")
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
        default="hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        help="Huggingface model name for Hunyuan-Image"
    )
    parser.add_argument(
        "--use-distilled",
        action="store_true",
        help="Use the distilled model variant for faster inference"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/hunyuan_image",
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
        help="Number of inference steps (use 8 for distilled model)"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for guidance"
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
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale for CFG (used via guider)"
    )
    parser.add_argument(
        "--distilled-guidance-scale",
        type=float,
        default=3.25,
        help="Guidance scale for distilled model"
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
        help="Batch size for generation"
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
    return parser.parse_args()


def load_model(args) -> HunyuanImagePipeline:
    """Load Hunyuan-Image 2.1 pipeline."""
    model_name = args.model
    if args.use_distilled:
        model_name = "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers"

    print(f"Loading model: {model_name}")
    pipe = HunyuanImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    # Update guider configuration for non-distilled model
    if not args.use_distilled and hasattr(pipe, 'guider'):
        pipe.guider = pipe.guider.new(guidance_scale=args.guidance_scale)
        print(f"Updated guider with guidance_scale={args.guidance_scale}")

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
        enhanced_prompts = {
            i: item.get("enhanced_prompt", item.get("prompt"))
            for i, item in enumerate(enhanced_data)
        }

    return metadatas, enhanced_prompts


def generate_images(
    pipe: HunyuanImagePipeline,
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

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": args.negative_prompt if args.negative_prompt else None,
            "height": args.H,
            "width": args.W,
            "num_inference_steps": args.steps,
            "num_images_per_prompt": current_batch,
            "generator": generator,
        }

        # Add distilled guidance scale for distilled model
        if args.use_distilled:
            gen_kwargs["distilled_guidance_scale"] = args.distilled_guidance_scale

        images = pipe(**gen_kwargs).images
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


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Hunyuan-Image generation")

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Load model
    pipe = load_model(args)

    # Load prompts
    metadatas, enhanced_prompts = load_prompts(args.metadata_file, args.enhanced_prompts)

    # Determine range
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(metadatas)

    print(f"Generating images for prompts {start_idx} to {end_idx - 1}")
    print(f"Output directory: {args.outdir}")
    print(f"Using enhanced prompts: {args.enhanced_prompts is not None}")
    print(f"Using distilled model: {args.use_distilled}")

    for index in trange(start_idx, end_idx, desc="Generating"):
        metadata = metadatas[index]
        seed_everything(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)

        # Determine which prompt to use
        if enhanced_prompts and index in enhanced_prompts:
            prompt = enhanced_prompts[index]
        else:
            prompt = metadata["prompt"]

        outpath = os.path.join(args.outdir, f"{index:05d}")

        # Skip if already generated
        if os.path.exists(os.path.join(outpath, "metadata.jsonl")):
            print(f"Skipping {index}: already exists")
            continue

        print(f"Prompt ({index:>3}/{len(metadatas)}): '{prompt}'")

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

    print("Done.")


if __name__ == "__main__":
    main()
