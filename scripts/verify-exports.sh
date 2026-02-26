#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GLUE="${1:-${PROJECT_DIR}/wasm/lightgbm.cjs}"

if [ ! -f "$GLUE" ]; then
  echo "ERROR: glue file not found: $GLUE"
  exit 1
fi

REQUIRED_SYMBOLS=(
  wl_lgb_get_last_error
  wl_lgb_dataset_create_from_mat
  wl_lgb_dataset_set_field
  wl_lgb_dataset_free
  wl_lgb_booster_create
  wl_lgb_booster_update
  wl_lgb_booster_get_num_classes
  wl_lgb_booster_free
  wl_lgb_booster_predict
  wl_lgb_booster_save_model
  wl_lgb_booster_load_model
)

missing=0
for fn in "${REQUIRED_SYMBOLS[@]}"; do
  if ! grep -q "_${fn}" "$GLUE"; then
    echo "MISSING: ${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} symbol(s) missing"
  exit 1
fi

echo "All ${#REQUIRED_SYMBOLS[@]} exports verified"
