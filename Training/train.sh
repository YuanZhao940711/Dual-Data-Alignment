#!/usr/bin/env bash
# Training Script for DINOv2-LoRA with Data Augmentation
# Usage:
#   bash train.sh -g 0 -c "RGB" -a 8 -n "experiment_name"

set -euo pipefail

# ========= User Configuration =========
# TODO: Update these paths for your environment
REAL_PATH="/root/autodl-tmp/AIGC_Detection/车损AIGC检测图片/重建汽车图片/real"
FAKE_PATH="/root/autodl-tmp/AIGC_Detection/车损AIGC检测图片/重建汽车图片/sd2.0"
QUALITY_JSON="./Training/MSCOCO_train2017.json"
CHECKPOINTS_ROOT="./checkpoints"

# Experiment Settings
LORA_RANK=8
LORA_ALPHA=1
OPTIM="adam"
NITER=1
BATCH_SIZE=16
ACCUM_STEPS=4
CROP_SIZE=336
LEARNING_RATE=1e-4

# Augmentation Settings
P_PIXELMIX=0.2
R_PIXELMIX=0.8
P_FREQMIX=0.2
R_FREQMIX=0.8

# ========= Command Line Arguments =========
GPU_ID=0
EXP_SUFFIX=""
RESUME=0

while getopts ":g:c:a:n:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG" ;;
    a) ACCUM_STEPS="$OPTARG" ;;
    n) EXP_SUFFIX="$OPTARG" ;;
  esac
done

# ========= Setup Flags & Name =========
EXP_NAME="DINO_${CROP_SIZE}_LoRA${LORA_RANK}_LR${LEARNING_RATE}_BS${BATCH_SIZE}"

if [[ -n "${EXP_SUFFIX}" ]]; then
  EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
fi

echo ">>> Starting Training: ${EXP_NAME}"
echo ">>> GPU: ${GPU_ID} | Accumulation: ${ACCUM_STEPS}"

BASE_DIR="${CHECKPOINTS_ROOT}/${EXP_NAME}"

mkdir -p "${BASE_DIR}"

if [[ "${RESUME}" == "1" ]]; then
    EXP_DIR=$(ls -td ${BASE_DIR}/* | head -1)
    echo ">>> Resuming from ${EXP_DIR}"
else
    TIME=$(date +"%Y%m%d_%H%M%S")
    EXP_DIR="${BASE_DIR}/${TIME}"
    mkdir -p "${EXP_DIR}"
    echo ">>> Starting new experiment: ${EXP_DIR}"
fi


# EXP_NAME="DINO_${CROP_SIZE}_LoRA${LORA_RANK}_LR${LEARNING_RATE}_BS${BATCH_SIZE}"
# if [[ -n "${EXP_SUFFIX}" ]]; then
#   EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
# fi

# echo ">>> Starting Training: ${EXP_NAME}"
# echo ">>> GPU: ${GPU_ID} | Accumulation: ${ACCUM_STEPS}"

python Training/train.py \
  --gpu_ids "${GPU_ID}" \
  --name "${EXP_NAME}" \
  --cropSize "${CROP_SIZE}" \
  --real_image_dir "${REAL_PATH}" \
  --vae_image_dir "${FAKE_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --accumulation_steps "${ACCUM_STEPS}" \
  --optim "${OPTIM}" \
  --niter "${NITER}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --checkpoints_dir "${EXP_DIR}" \
  --quality_json "${QUALITY_JSON}" \
  --p_pixelmix "${P_PIXELMIX}" \
  --r_pixelmix "${R_PIXELMIX}" \
  --p_freqmix "${P_FREQMIX}" \
  --r_freqmix "${R_FREQMIX}" \
  --seed 42 \
  --use_amp \
  --resume "${RESUME}"

echo ">>> Training finished. Checkpoints saved to: ${EXP_DIR}"
