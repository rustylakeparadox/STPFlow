#!/bin/bash

set -u
set -o pipefail

# =========================
# Batch evaluation for STPFlow hyperparameter experiments
# =========================

ROOT_DIR="/home/zhangdaoliang/liuwy/STPFlow-main/results_hparam"
EVAL_SCRIPT="/home/zhangdaoliang/liuwy/STPFlow-main/results/evaluation.py"

LOG_FILE="${ROOT_DIR}/batch_hparam_evaluation.log"
SUMMARY_INDEX="${ROOT_DIR}/hparam_evaluated_files.tsv"

# =========================
# Four transfer settings
# =========================

PARENT_EXPS=(
    "GSE198353_1to2"
    "GSE198353_2to1"
    "spatialCOC_2to3"
    "spatialCOC_3to2"
)

# =========================
# Hyperparameter experiment folders
# Keep this order for later plotting
# =========================

PARAM_EXPS=(
    "default"

    "flow_steps_1"
    "flow_steps_10"
    "flow_steps_50"

    "prior_noise_0.1"
    "prior_noise_0.3"

    "prior_weight_0"
    "prior_weight_0.5"

    "uce_dropout_aug_0"
    "uce_dropout_aug_0.3"
    "uce_dropout_aug_0.5"
)

# =========================
# Initialize log
# =========================

echo "======================================" | tee "$LOG_FILE"
echo "Start hyperparameter batch evaluation" | tee -a "$LOG_FILE"
echo "Root dir: $ROOT_DIR" | tee -a "$LOG_FILE"
echo "Eval script: $EVAL_SCRIPT" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

echo -e "parent_experiment\tparameter_experiment\tnpz_path\tmetrics_path" > "$SUMMARY_INDEX"

if [ ! -d "$ROOT_DIR" ]; then
    echo "[ERROR] ROOT_DIR not found: $ROOT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[ERROR] evaluation.py not found: $EVAL_SCRIPT" | tee -a "$LOG_FILE"
    exit 1
fi

# =========================
# Function: find best_predictions.npz
# =========================

find_npz() {
    local TARGET_DIR=$1

    # 1. Most common locations
    if [ -f "${TARGET_DIR}/best_predictions.npz" ]; then
        echo "${TARGET_DIR}/best_predictions.npz"
        return
    fi

    if [ -f "${TARGET_DIR}/save/best_predictions.npz" ]; then
        echo "${TARGET_DIR}/save/best_predictions.npz"
        return
    fi

    if [ -f "${TARGET_DIR}/results/best_predictions.npz" ]; then
        echo "${TARGET_DIR}/results/best_predictions.npz"
        return
    fi

    # 2. Recursive search: choose the most recently modified one
    local FOUND
    FOUND=$(find "$TARGET_DIR" -type f -name "best_predictions.npz" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)

    if [ -n "$FOUND" ]; then
        echo "$FOUND"
    fi
}

# =========================
# Function: run evaluation
# =========================

run_eval() {
    local PARENT_NAME=$1
    local PARAM_NAME=$2
    local TARGET_DIR=$3

    if [ ! -d "$TARGET_DIR" ]; then
        echo "[SKIP] Folder not found: $TARGET_DIR" | tee -a "$LOG_FILE"
        return
    fi

    local NPZ_PATH
    NPZ_PATH=$(find_npz "$TARGET_DIR")

    if [ -z "${NPZ_PATH:-}" ]; then
        echo "[SKIP] No best_predictions.npz found in $TARGET_DIR" | tee -a "$LOG_FILE"
        return
    fi

    local NPZ_DIR
    NPZ_DIR=$(dirname "$NPZ_PATH")

    echo "" | tee -a "$LOG_FILE"
    echo "[RUN] parent=${PARENT_NAME}, param=${PARAM_NAME}" | tee -a "$LOG_FILE"
    echo "Target dir: $TARGET_DIR" | tee -a "$LOG_FILE"
    echo "NPZ path:   $NPZ_PATH" | tee -a "$LOG_FILE"
    echo "NPZ dir:    $NPZ_DIR" | tee -a "$LOG_FILE"

    (
        cd "$NPZ_DIR" || exit 1

        python "$EVAL_SCRIPT" \
            --npz best_predictions.npz \
            --out metrics.json \
            --per_protein_out per_protein_metrics.csv \
            --per_cell_out per_cell_metrics.csv
    ) 2>&1 | tee -a "$LOG_FILE"

    local STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -eq 0 ]; then
        echo "[DONE] parent=${PARENT_NAME}, param=${PARAM_NAME}" | tee -a "$LOG_FILE"
        echo -e "${PARENT_NAME}\t${PARAM_NAME}\t${NPZ_PATH}\t${NPZ_DIR}/metrics.json" >> "$SUMMARY_INDEX"
    else
        echo "[FAILED] parent=${PARENT_NAME}, param=${PARAM_NAME}" | tee -a "$LOG_FILE"
    fi
}

# =========================
# Main loop
# =========================

for PARENT_NAME in "${PARENT_EXPS[@]}"; do

    PARENT_DIR="${ROOT_DIR}/${PARENT_NAME}"

    echo "" | tee -a "$LOG_FILE"
    echo "======================================" | tee -a "$LOG_FILE"
    echo "[PARENT EXPERIMENT] $PARENT_NAME" | tee -a "$LOG_FILE"
    echo "Path: $PARENT_DIR" | tee -a "$LOG_FILE"
    echo "======================================" | tee -a "$LOG_FILE"

    if [ ! -d "$PARENT_DIR" ]; then
        echo "[SKIP] Parent folder not found: $PARENT_DIR" | tee -a "$LOG_FILE"
        continue
    fi

    for PARAM_NAME in "${PARAM_EXPS[@]}"; do
        TARGET_DIR="${PARENT_DIR}/${PARAM_NAME}"
        run_eval "$PARENT_NAME" "$PARAM_NAME" "$TARGET_DIR"
    done

done

echo "" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "Hyperparameter batch evaluation finished." | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Evaluated file index saved to: $SUMMARY_INDEX" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"