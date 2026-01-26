#!/bin/bash
#
# Multi-GPU GenEval Generation Launcher
#
# Usage:
#   ./run_multi_gpu.sh qwen 4 /path/to/model outputs/qwen_original [--batch_size 2]
#   ./run_multi_gpu.sh hunyuan 8 /path/to/model outputs/hunyuan_original [--use-distilled] [--batch_size 2]
#

set -e

MODEL_TYPE="${1:-qwen}"
NUM_GPUS="${2:-1}"
MODEL_PATH="${3}"
OUTDIR="${4:-outputs}"
shift 4 || true

# Parse optional arguments
BATCH_SIZE=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METADATA_FILE="$SCRIPT_DIR/../prompts/evaluation_metadata.jsonl"

# Get total prompt count dynamically
TOTAL_PROMPTS=$(wc -l < "$METADATA_FILE" 2>/dev/null | tr -d ' ')
if [[ -z "$TOTAL_PROMPTS" || "$TOTAL_PROMPTS" -eq 0 ]]; then
    echo "ERROR: Cannot read metadata file: $METADATA_FILE"
    exit 1
fi

echo "============================================"
echo "Multi-GPU GenEval Generation"
echo "============================================"
echo "Model type: $MODEL_TYPE"
echo "Number of GPUs: $NUM_GPUS"
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTDIR"
echo "Batch size: $BATCH_SIZE"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "============================================"

# Determine which script to use
if [[ "$MODEL_TYPE" == "qwen" ]]; then
    SCRIPT="$SCRIPT_DIR/qwen_image_generate.py"
elif [[ "$MODEL_TYPE" == "hunyuan" ]]; then
    SCRIPT="$SCRIPT_DIR/hunyuan_image_generate.py"
else
    echo "Unknown model type: $MODEL_TYPE (use 'qwen' or 'hunyuan')"
    exit 1
fi

# Create output directory
mkdir -p "$OUTDIR"

# Launch processes for each GPU
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    echo "Launching GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python "$SCRIPT" "$METADATA_FILE" \
        --model "$MODEL_PATH" \
        --outdir "$OUTDIR" \
        --gpu-id $i \
        --num-gpus $NUM_GPUS \
        --batch_size $BATCH_SIZE \
        "${EXTRA_ARGS[@]}" \
        > "$OUTDIR/gpu_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${NUM_GPUS} processes: ${PIDS[@]}"
echo "Logs: $OUTDIR/gpu_*.log"
echo ""
echo "Monitoring progress (Ctrl+C to detach, processes continue)..."
echo ""

# Monitor progress
while true; do
    # Count completed folders
    if [[ -d "$OUTDIR" ]]; then
        COMPLETED=$(find "$OUTDIR" -name "metadata.jsonl" 2>/dev/null | wc -l)
        echo -ne "\rCompleted: $COMPLETED / $TOTAL_PROMPTS prompts"
    fi

    # Check if all processes are still running
    ALL_DONE=true
    for PID in "${PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            ALL_DONE=false
        fi
    done

    if $ALL_DONE; then
        echo ""
        break
    fi

    sleep 5
done

# Check exit codes
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" 2>/dev/null || FAILED=$((FAILED + 1))
done

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: $FAILED process(es) failed. Check logs in $OUTDIR/gpu_*.log"
fi

echo ""
echo "============================================"
echo "Generation complete!"
echo "Total: $(find "$OUTDIR" -name "metadata.jsonl" | wc -l) / $TOTAL_PROMPTS"
if [[ $FAILED -gt 0 ]]; then
    echo "Failed processes: $FAILED"
fi
echo "============================================"
