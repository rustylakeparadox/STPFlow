#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/home/zhangdaoliang/liuwy/STPFlow-main"
TRAIN_SCRIPT="${PROJECT_ROOT}/ST-FlowPro/train.py"

HPARAM_ROOT="${PROJECT_ROOT}/results_hparam"
mkdir -p "${HPARAM_ROOT}"

GPU_ID=0

PROTEIN_VAE_CKPT="/home/zhangdaoliang/liuwy/STPFlow-main/results/spatialCOC_3to2/protein_vae.pt"
FIXED_RESULT_ROOT="/home/zhangdaoliang/liuwy/STPFlow-main/results/spatialCOC_3to2"

# =========================
# Default hyperparameters
# =========================
DEFAULT_N_STEPS=20
DEFAULT_PRIOR_NOISE=0.0
DEFAULT_PRIOR_WEIGHT=0.1
DEFAULT_UCE_DROPOUT_AUG=0.1

BASE_ARGS=(
    --train_rna_path /home/zhangdaoliang/liuwy/benchmark_data/SpatialCOC/Mouse_Thymus_3/adata3_RNA.h5ad
    --train_adt_path /home/zhangdaoliang/liuwy/benchmark_data/SpatialCOC/Mouse_Thymus_3/adata3_ADT.h5ad
    --test_rna_path /home/zhangdaoliang/liuwy/benchmark_data/SpatialCOC/Mouse_Thymus_2/adata2_RNA.h5ad
    --test_adt_path /home/zhangdaoliang/liuwy/benchmark_data/SpatialCOC/Mouse_Thymus_2/adata2_ADT.h5ad

    --device 0
    --epochs 300
    --batch_size 256
    --lr 1e-3
    --hidden_dim 256

    --use_uce
    --species mouse
)

run_exp () {
    EXP_NAME=$1
    N_STEPS=$2
    PRIOR_NOISE=$3
    PRIOR_WEIGHT=$4
    UCE_DROPOUT_AUG=$5

    EXP_ROOT="${HPARAM_ROOT}/${EXP_NAME}"
    SAVE_DIR="${EXP_ROOT}/save"

    mkdir -p "${SAVE_DIR}"

    echo "======================================"
    echo "[RUN] ${EXP_NAME}"
    echo "save_dir               = ${SAVE_DIR}"
    echo "fixed_result_root      = ${FIXED_RESULT_ROOT}"
    echo "protein_vae_ckpt       = ${PROTEIN_VAE_CKPT}"
    echo "n_steps                = ${N_STEPS}"
    echo "prior_noise_scale_eval = ${PRIOR_NOISE}"
    echo "prior_weight           = ${PRIOR_WEIGHT}"
    echo "uce_dropout_aug        = ${UCE_DROPOUT_AUG}"
    echo "======================================"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${TRAIN_SCRIPT}" \
        "${BASE_ARGS[@]}" \
        --protein_vae_ckpt "${PROTEIN_VAE_CKPT}" \
        --save_dir "${SAVE_DIR}" \
        --result_root "${FIXED_RESULT_ROOT}" \
        --n_steps "${N_STEPS}" \
        --prior_noise_scale_eval "${PRIOR_NOISE}" \
        --prior_weight "${PRIOR_WEIGHT}" \
        --uce_dropout_aug "${UCE_DROPOUT_AUG}" \
        > "${EXP_ROOT}/train.log" 2>&1

    echo "[DONE] ${EXP_NAME}"
    echo ""
}

# =========================
# Experiments
# =========================

run_exp "default" \
    ${DEFAULT_N_STEPS} \
    ${DEFAULT_PRIOR_NOISE} \
    ${DEFAULT_PRIOR_WEIGHT} \
    ${DEFAULT_UCE_DROPOUT_AUG}

# Flow refinement steps
run_exp "flow_steps_1" \
    1 \
    ${DEFAULT_PRIOR_NOISE} \
    ${DEFAULT_PRIOR_WEIGHT} \
    ${DEFAULT_UCE_DROPOUT_AUG}

run_exp "flow_steps_10" \
    10 \
    ${DEFAULT_PRIOR_NOISE} \
    ${DEFAULT_PRIOR_WEIGHT} \
    ${DEFAULT_UCE_DROPOUT_AUG}

run_exp "flow_steps_50" \
    50 \
    ${DEFAULT_PRIOR_NOISE} \
    ${DEFAULT_PRIOR_WEIGHT} \
    ${DEFAULT_UCE_DROPOUT_AUG}

# Conditional prior noise scale
run_exp "prior_noise_0.1" \
    ${DEFAULT_N_STEPS} \
    0.1 \
    ${DEFAULT_PRIOR_WEIGHT} \
    ${DEFAULT_UCE_DROPOUT_AUG}

run_exp "prior_noise_0.3" \
    ${DEFAULT_N_STEPS} \
    0.3 \
    ${DEFAULT_PRIOR_WEIGHT} \
    ${DEFAULT_UCE_DROPOUT_AUG}

# Prior regularization weight
run_exp "prior_weight_0" \
    ${DEFAULT_N_STEPS} \
    ${DEFAULT_PRIOR_NOISE} \
    0.0 \
    ${DEFAULT_UCE_DROPOUT_AUG}

run_exp "prior_weight_0.5" \
    ${DEFAULT_N_STEPS} \
    ${DEFAULT_PRIOR_NOISE} \
    0.5 \
    ${DEFAULT_UCE_DROPOUT_AUG}

# UCE dropout augmentation
run_exp "uce_dropout_aug_0" \
    ${DEFAULT_N_STEPS} \
    ${DEFAULT_PRIOR_NOISE} \
    ${DEFAULT_PRIOR_WEIGHT} \
    0.0

run_exp "uce_dropout_aug_0.3" \
    ${DEFAULT_N_STEPS} \
    ${DEFAULT_PRIOR_NOISE} \
    ${DEFAULT_PRIOR_WEIGHT} \
    0.3

run_exp "uce_dropout_aug_0.5" \
    ${DEFAULT_N_STEPS} \
    ${DEFAULT_PRIOR_NOISE} \
    ${DEFAULT_PRIOR_WEIGHT} \
    0.5

echo "All hyperparameter experiments finished."