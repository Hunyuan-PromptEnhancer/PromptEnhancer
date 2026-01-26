#!/bin/bash
#
# GenEval Generation Runner
# Runs both original and enhanced prompt generation for specified model
#
# Usage:
#   ./run_geneval.sh qwen [--enhanced-prompts enhanced.jsonl]
#   ./run_geneval.sh hunyuan [--distilled] [--enhanced-prompts enhanced.jsonl]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTS_DIR="$SCRIPT_DIR/../prompts"
METADATA_FILE="$PROMPTS_DIR/evaluation_metadata.jsonl"

MODEL_TYPE="${1:-qwen}"
shift || true

# Default parameters
N_SAMPLES=4
STEPS=50
HEIGHT=1024
WIDTH=1024
SEED=42

# Parse arguments
ENHANCED_PROMPTS=""
USE_DISTILLED=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --enhanced-prompts)
            ENHANCED_PROMPTS="$2"
            shift 2
            ;;
        --distilled)
            USE_DISTILLED="--use-distilled"
            STEPS=8
            shift
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "============================================"
echo "GenEval Image Generation"
echo "============================================"
echo "Model: $MODEL_TYPE"
echo "Metadata: $METADATA_FILE"
echo "Samples per prompt: $N_SAMPLES"
echo "Inference steps: $STEPS"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "============================================"

if [[ "$MODEL_TYPE" == "qwen" ]]; then
    SCRIPT="$SCRIPT_DIR/qwen_image_generate.py"
    BASE_OUTDIR="outputs/qwen_image"

    # Generate with original prompts
    echo ""
    echo ">>> Generating with ORIGINAL prompts..."
    python "$SCRIPT" "$METADATA_FILE" \
        --outdir "${BASE_OUTDIR}_original" \
        --n_samples $N_SAMPLES \
        --steps $STEPS \
        --H $HEIGHT \
        --W $WIDTH \
        --seed $SEED \
        $EXTRA_ARGS

    # Generate with enhanced prompts if provided
    if [[ -n "$ENHANCED_PROMPTS" ]]; then
        echo ""
        echo ">>> Generating with ENHANCED prompts..."
        python "$SCRIPT" "$METADATA_FILE" \
            --enhanced-prompts "$ENHANCED_PROMPTS" \
            --outdir "${BASE_OUTDIR}_enhanced" \
            --n_samples $N_SAMPLES \
            --steps $STEPS \
            --H $HEIGHT \
            --W $WIDTH \
            --seed $SEED \
            $EXTRA_ARGS
    fi

elif [[ "$MODEL_TYPE" == "hunyuan" ]]; then
    SCRIPT="$SCRIPT_DIR/hunyuan_image_generate.py"
    BASE_OUTDIR="outputs/hunyuan_image"

    if [[ -n "$USE_DISTILLED" ]]; then
        BASE_OUTDIR="${BASE_OUTDIR}_distilled"
    fi

    # Generate with original prompts
    echo ""
    echo ">>> Generating with ORIGINAL prompts..."
    python "$SCRIPT" "$METADATA_FILE" \
        --outdir "${BASE_OUTDIR}_original" \
        --n_samples $N_SAMPLES \
        --steps $STEPS \
        --H $HEIGHT \
        --W $WIDTH \
        --seed $SEED \
        $USE_DISTILLED \
        $EXTRA_ARGS

    # Generate with enhanced prompts if provided
    if [[ -n "$ENHANCED_PROMPTS" ]]; then
        echo ""
        echo ">>> Generating with ENHANCED prompts..."
        python "$SCRIPT" "$METADATA_FILE" \
            --enhanced-prompts "$ENHANCED_PROMPTS" \
            --outdir "${BASE_OUTDIR}_enhanced" \
            --n_samples $N_SAMPLES \
            --steps $STEPS \
            --H $HEIGHT \
            --W $WIDTH \
            --seed $SEED \
            $USE_DISTILLED \
            $EXTRA_ARGS
    fi

else
    echo "Unknown model type: $MODEL_TYPE"
    echo "Supported: qwen, hunyuan"
    exit 1
fi

echo ""
echo "============================================"
echo "Generation complete!"
echo "============================================"
