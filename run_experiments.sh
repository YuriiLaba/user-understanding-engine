#!/usr/bin/env bash
# Experiment commands, run top to bottom for each model. Comment out a block with '#' to skip it.

# Stop on the first failed run. pipefail makes a failure propagate through the
# `tee` pipe (otherwise the exit code would be tee's, which is always 0).
set -eo pipefail

# CUDA 13.0 toolchain + SGLang flags (needed by every run).
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"
export SGLANG_DISABLE_CUDNN_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# Timestamped progress line.
log() { echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# All models to run AX and metadata experiments with.
MODELS=(
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "google/gemma-4-E4B-it"
    "LiquidAI/LFM2.5-VL-1.6B"
    "LiquidAI/LFM2.5-8B-A1B"
    "ibm-granite/granite-4.1-8b"
    "ibm-granite/granite-4.1-3b"
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    "LiquidAI/LFM2.5-8B-A1B"
)

# Vision-capable models — screen experiments (pure + OCR) run only for these.
MULTIMODAL_MODELS=(
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "google/gemma-4-E4B-it"
    "LiquidAI/LFM2.5-VL-1.6B"
)

is_multimodal() {
    local model="$1"
    for mm in "${MULTIMODAL_MODELS[@]}"; do
        [[ "$mm" == "$model" ]] && return 0
    done
    return 1
}

for MODEL in "${MODELS[@]}"; do
    SLUG=${MODEL//\//_}   # filesystem-safe model name for log filenames

    echo
    log "============================================================"
    log "MODEL: $MODEL"
    log "============================================================"

    if is_multimodal "$MODEL"; then
        # --- Screen (pure vision, no OCR) ---
        log "[$MODEL] running: screen"
        uv run --no-sync python main_screen.py \
            --user-name hlib \
            --model-name "$MODEL" \
            --no-apply-ocr \
            --paths-to-videos \
            raw_data/hlib/2025-11-28_15-00-53.mp4 \
            raw_data/hlib/2025-11-28_15-30-53.mp4 \
            raw_data/hlib/2025-11-28_16-00-53.mp4 \
            raw_data/hlib/2025-11-28_16-30-54.mp4 \
            raw_data/hlib/2025-11-28_17-00-54.mp4 \
            raw_data/hlib/2025-11-28_17-30-54.mp4 \
            raw_data/hlib/2025-11-28_18-00-54.mp4 \
            2>&1 | tee "logs/${SLUG}_screen.log"
    fi

    # --- Screen (OCR) ---
    log "[$MODEL] running: ocr"
    uv run --no-sync python main_screen.py \
        --user-name hlib \
        --model-name "$MODEL" \
        --apply-ocr \
        --experiment-prefix ocr \
        --paths-to-videos \
        raw_data/hlib/2025-11-28_15-00-53.mp4 \
        raw_data/hlib/2025-11-28_15-30-53.mp4 \
        raw_data/hlib/2025-11-28_16-00-53.mp4 \
        raw_data/hlib/2025-11-28_16-30-54.mp4 \
        raw_data/hlib/2025-11-28_17-00-54.mp4 \
        raw_data/hlib/2025-11-28_17-30-54.mp4 \
        raw_data/hlib/2025-11-28_18-00-54.mp4 \
        2>&1 | tee "logs/${SLUG}_ocr.log"

    # --- Accessibility (AX) ---
    log "[$MODEL] running: ax"
    uv run --no-sync python main_ax.py \
        --user-name hlib \
        --model-name "$MODEL" \
        --path-to-accessibility raw_data/hlib/accessibility.json \
        2>&1 | tee "logs/${SLUG}_ax.log"

    # --- Metadata ---
    log "[$MODEL] running: metadata"
    uv run --no-sync python main_metadata.py \
        --user-name hlib \
        --model-name "$MODEL" \
        --path-to-metadata raw_data/hlib \
        2>&1 | tee "logs/${SLUG}_metadata.log"
done
