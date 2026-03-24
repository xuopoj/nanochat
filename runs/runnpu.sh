#!/bin/bash

# Offline training run for Ascend NPU (Atlas 800T A2 or similar).
# Assumes all data and model assets are pre-downloaded and mounted into the container.
# Run download_offline.py on an online machine first, then mount the cache dirs here.

# Usage:
#   bash runs/runnpu.sh
# With wandb logging:
#   WANDB_RUN=npu-run bash runs/runnpu.sh

# -----------------------------------------------------------------------------
# Mount paths — adjust these to match your container/node mount points
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/nanochat}"
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_climbmix}"

# Offline mode: disable all HuggingFace network calls
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/huggingface/datasets}"

export OMP_NUM_THREADS=1
mkdir -p $NANOCHAT_BASE_DIR

# All Python deps are pre-installed in the Docker image — no venv needed.
# Locate torchrun (pip may install it under ~/.local/bin instead of /usr/bin)
TORCHRUN=$(find /home/ma-user/.local /usr/local/bin /usr/bin -name torchrun 2>/dev/null | head -1)
[ -z "$TORCHRUN" ] && TORCHRUN=$(python -c "import sysconfig,os; print(os.path.join(sysconfig.get_path('scripts'),'torchrun'))")
echo "Using torchrun: $TORCHRUN"

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Number of NPU devices (adjust to your node configuration)
NPROC=${NPROC_PER_NODE:-8}

# -----------------------------------------------------------------------------
# Tokenizer (uses pre-downloaded data, no network calls)

python -m scripts.tok_train
# tok_eval compares against GPT-2/GPT-4 tiktoken encodings which require network access.
# Skip it in offline environments — it's informational only, not required for training.

# -----------------------------------------------------------------------------
# Base model pretraining on NPU

$TORCHRUN --standalone --nproc_per_node=$NPROC -m scripts.base_train \
    --device-type=npu \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --window-pattern=L \
    --run=$WANDB_RUN

$TORCHRUN --standalone --nproc_per_node=$NPROC -m scripts.base_eval \
    --device-type=npu \
    --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (identity conversations must be pre-downloaded into NANOCHAT_BASE_DIR)

$TORCHRUN --standalone --nproc_per_node=$NPROC -m scripts.chat_sft \
    --device-type=npu \
    --device-batch-size=16 \
    --run=$WANDB_RUN

$TORCHRUN --standalone --nproc_per_node=$NPROC -m scripts.chat_eval \
    --device-type=npu \
    -i sft
