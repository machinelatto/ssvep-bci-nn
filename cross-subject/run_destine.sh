#!/usr/bin/env bash
set -euo pipefail

# Run all DESTINE experiments with consistent settings.
# Optional first arg lets you set a custom run id.
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

USERS="subject_01,subject_02,subject_03,subject_04,subject_05,subject_06,subject_07,subject_08,subject_09,subject_10"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMMON_ARGS=(
  --users "$USERS"
  --frequencies "12,15,20"
  --window 0.4
  --loader-window-mode multiple
  --loader-window-overlap 0
  --sample-rate 256
  --car-mode global
  --no-bandpass # ja tem filtro
)

OPT_ARGS=(
  --learning-rate 1e-3
  --weight-decay 0.0001
  --batch-size 256
)

echo "[${RUN_ID}] Starting DESTINE CCA"
python run_destine_cca_experiments.py \
  "${COMMON_ARGS[@]}" \
  --num-harmonics 3 \
  --results-dir "destine_results_multiple_norm_reg/cca/"

echo "[${RUN_ID}] Starting DESTINE EEGNet"
python run_destine_eegnet_experiments.py \
  "${COMMON_ARGS[@]}" \
  "${OPT_ARGS[@]}" \
  --results-dir "destine_results_multiple_norm_reg/eegnet/"

echo "[${RUN_ID}] Starting DESTINE EEGNet+CCA (per-frequency)"
python run_destine_eegnetcca_experiments.py \
  "${COMMON_ARGS[@]}" \
  "${OPT_ARGS[@]}" \
  --num-harmonics 3 \
  --cca-mode per-frequency \
  --results-dir "destine_results_multiple_norm_reg/eegnetcca/"

echo "[${RUN_ID}] Starting DESTINE EEGNet+FBCCA per-frequency"
python run_destine_eegnetfbcca_experiments.py \
  "${COMMON_ARGS[@]}" \
  "${OPT_ARGS[@]}" \
  --num-harmonics 3 \
  --results-dir "destine_results_multiple_norm_reg/eegnetfbcca/" \
  --fbcca-mode per-frequency

echo "[${RUN_ID}] All DESTINE runs completed."
