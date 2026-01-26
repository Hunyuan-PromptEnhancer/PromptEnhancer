#!/bin/bash
#
# Multi-GPU Prompt Enhancement Launcher
#
# Usage:
#   ./run_enhance_prompts.sh 4 /path/to/prompt_enhancer_model
#

set -e

NUM_GPUS="${1:-1}"
MODEL_PATH="${2}"
OUTPUT_FILE="${3:-../prompts/enhanced_prompts.jsonl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FILE="$SCRIPT_DIR/../prompts/evaluation_metadata.jsonl"

# Get total prompt count dynamically
TOTAL=$(wc -l < "$INPUT_FILE" 2>/dev/null | tr -d ' ')
if [[ -z "$TOTAL" || "$TOTAL" -eq 0 ]]; then
    echo "ERROR: Cannot read input file: $INPUT_FILE"
    exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 <num_gpus> <model_path> [output_file]"
    echo "Example: $0 4 /path/to/prompt_enhancer_model"
    exit 1
fi

echo "============================================"
echo "Multi-GPU Prompt Enhancement"
echo "============================================"
echo "Number of GPUs: $NUM_GPUS"
echo "Model path: $MODEL_PATH"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "============================================"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Clear output file if starting fresh
# Comment out next line to enable resume
# rm -f "$OUTPUT_FILE" "${OUTPUT_FILE}.lock"

# Launch processes for each GPU
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    echo "Launching GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python "$SCRIPT_DIR/batch_enhance_prompts.py" \
        --model "$MODEL_PATH" \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --gpu-id $i \
        --num-gpus $NUM_GPUS \
        > "enhance_gpu_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${NUM_GPUS} processes: ${PIDS[@]}"
echo "Logs: enhance_gpu_*.log"
echo ""
echo "Monitoring progress (Ctrl+C to detach, processes continue)..."

# Monitor progress
while true; do
    if [[ -f "$OUTPUT_FILE" ]]; then
        COMPLETED=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
        echo -ne "\rCompleted: $COMPLETED / $TOTAL prompts"
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

    sleep 3
done

# Check exit codes
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" 2>/dev/null || FAILED=$((FAILED + 1))
done

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: $FAILED process(es) failed. Check logs: enhance_gpu_*.log"
fi

# Sort output by index for consistent ordering (with file lock for safety)
if [[ -f "$OUTPUT_FILE" ]]; then
    echo ""
    echo "Sorting output by index..."
    LOCK_FILE="${OUTPUT_FILE}.lock"
    python3 -c "
import json
import filelock
import sys

output_file = '$OUTPUT_FILE'
lock_file = '$LOCK_FILE'

lock = filelock.FileLock(lock_file, timeout=60)
try:
    with lock:
        with open(output_file) as f:
            lines = f.readlines()

        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'Warning: Skipping malformed line {i+1}: {e}', file=sys.stderr)

        data.sort(key=lambda x: x.get('index', 0))

        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f'Sorted {len(data)} results')
except filelock.Timeout:
    print('ERROR: Could not acquire lock for sorting', file=sys.stderr)
    sys.exit(1)
"
fi

echo ""
echo "============================================"
echo "Enhancement complete!"
echo "Output: $OUTPUT_FILE"
if [[ $FAILED -gt 0 ]]; then
    echo "Failed processes: $FAILED"
fi
echo "============================================"
