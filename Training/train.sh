#!/usr/bin/env bash
# Training Script for DINOv2-LoRA with DDP support
#
# 单卡启动:
#   bash train.sh -g 0 -a 4 -n "experiment_name"
#
# 多卡 DDP 启动 (例如 4 卡):
#   bash train.sh -g 0,1,2,3 -a 1 -n "experiment_name"
#   注意：多卡时 -a (accumulation_steps) 可相应减小，因为 world_size 已扩大等效 batch

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

# ===== 低误报率控制（保险业务关键参数）=====
# fp_penalty_weight > 1：加重"把真实图判为AI生成"的惩罚，降低误报率（FPR）
# 车损图片 / 单证图片场景建议：3.0 ~ 5.0
# 值越大 → 误报率越低，但漏报率（FNR）会相应上升
FP_PENALTY=3.0

# 损失权重（cls + contrastive = 1.0 为宜）
LOSS_CLS_WEIGHT=0.5
LOSS_CONTRASTIVE_WEIGHT=0.5

# ========= Command Line Arguments =========
GPU_IDS="0"
EXP_SUFFIX=""
RESUME=0

while getopts ":g:a:n:r:" opt; do
  case $opt in
    g) GPU_IDS="$OPTARG" ;;
    a) ACCUM_STEPS="$OPTARG" ;;
    n) EXP_SUFFIX="$OPTARG" ;;
    r) RESUME="$OPTARG" ;;
  esac
done

# ========= 解析 GPU 数量 =====
# 将 "0,1,2,3" 转为数组，统计卡数
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

# ========= Setup Flags & Name =========
EXP_NAME="DINO_${CROP_SIZE}_LoRA${LORA_RANK}_LR${LEARNING_RATE}_BS${BATCH_SIZE}"
if [[ -n "${EXP_SUFFIX}" ]]; then
  EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
fi

echo ">>> Starting Training: ${EXP_NAME}"
echo ">>> GPU(s): ${GPU_IDS} (${NUM_GPUS} GPU(s)) | Accumulation: ${ACCUM_STEPS}"

BASE_DIR="${CHECKPOINTS_ROOT}/${EXP_NAME}"
mkdir -p "${BASE_DIR}"

if [[ "${RESUME}" == "1" ]]; then
    EXP_DIR=$(ls -td ${BASE_DIR}/* 2>/dev/null | head -1)
    if [[ -z "${EXP_DIR}" ]]; then
        echo ">>> No existing checkpoint dir found, starting new experiment."
        TIME=$(date +"%Y%m%d_%H%M%S")
        EXP_DIR="${BASE_DIR}/${TIME}"
        mkdir -p "${EXP_DIR}"
    else
        echo ">>> Resuming from ${EXP_DIR}"
    fi
else
    TIME=$(date +"%Y%m%d_%H%M%S")
    EXP_DIR="${BASE_DIR}/${TIME}"
    mkdir -p "${EXP_DIR}"
    echo ">>> Starting new experiment: ${EXP_DIR}"
fi

# ========= 公共训练参数 =========
COMMON_ARGS=(
  --name           "${EXP_NAME}"
  --cropSize       "${CROP_SIZE}"
  --real_image_dir "${REAL_PATH}"
  --vae_image_dir  "${FAKE_PATH}"
  --batch_size     "${BATCH_SIZE}"
  --lr             "${LEARNING_RATE}"
  --accumulation_steps "${ACCUM_STEPS}"
  --optim          "${OPTIM}"
  --niter          "${NITER}"
  --lora_rank      "${LORA_RANK}"
  --lora_alpha     "${LORA_ALPHA}"
  --checkpoints_dir "${EXP_DIR}"
  --quality_json   "${QUALITY_JSON}"
  --p_pixelmix     "${P_PIXELMIX}"
  --r_pixelmix     "${R_PIXELMIX}"
  --p_freqmix      "${P_FREQMIX}"
  --r_freqmix      "${R_FREQMIX}"
  --fp_penalty_weight "${FP_PENALTY}"
  --loss_cls_weight "${LOSS_CLS_WEIGHT}"
  --loss_contrastive_weight "${LOSS_CONTRASTIVE_WEIGHT}"
  --seed           42
  --use_amp
  --resume         "${RESUME}"
)

# ========= 启动训练 =========
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    echo ">>> Launching DDP with torchrun (${NUM_GPUS} GPUs)"
    # torchrun 会自动注入 LOCAL_RANK / RANK / WORLD_SIZE 环境变量
    # --gpu_ids 只传 "0" 作占位（DDP 模式下实际由 LOCAL_RANK 决定设备）
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        Training/train.py \
        --gpu_ids "0" \
        "${COMMON_ARGS[@]}"
else
    echo ">>> Launching single-GPU training"
    python Training/train.py \
        --gpu_ids "${GPU_IDS}" \
        "${COMMON_ARGS[@]}"
fi

echo ">>> Training finished. Checkpoints saved to: ${EXP_DIR}"
