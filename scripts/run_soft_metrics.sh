#!/usr/bin/env bash
#
# Calibrate the soft-coverage metric once on ALL users in a propositions folder, then score
# every (model, merge arm, modality set) with the calibrated bounds.
#
# Step 1  scripts/calibrate_soft_metric.py  -> <results>/soft_metric_calibration.json
#         (low/high = cosine where the reranker starts / fully agrees two propositions
#          state the same fact; see the script docstring for the rule)
# Step 2  scripts/compare_modalities_vs_ideal.py --skip-reranker --soft-low L --soft-high H
#         for each user x model x method x modality set  -> <results>/<user>/<model>/<method>__<mods>.csv
# Step 3  scripts/build_comparison.py  -> <results>/comparison.csv (+ .json/.html)
#
# Only embedding metrics are computed (no cross-encoder judge), so a full sweep is minutes,
# not hours. The embeddings_reranker_llm arm calls OpenAI for synthesis; drop it from
# METHODS if you do not want that cost.
#
# Usage:
#   ./scripts/run_soft_metrics.sh                       # new_data -> results_soft
#   PROP_ROOT=new_data RESULTS_ROOT=results_soft ./scripts/run_soft_metrics.sh
#   USERS="hlib pavlo" MODELS="qwen3_8b" ./scripts/run_soft_metrics.sh
#   SKIP_CALIBRATION=1 ./scripts/run_soft_metrics.sh    # reuse an existing calibration json
#   FORCE=1 ./scripts/run_soft_metrics.sh               # recompute CSVs that already exist
#
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- config -----------------------------------------------------------------
PYTHON="${PYTHON:-.venv/bin/python}"
PROP_ROOT="${PROP_ROOT:-new_data}"
RESULTS_ROOT="${RESULTS_ROOT:-results_soft}"
USERS="${USERS:-}"                 # space-separated; empty = every folder under PROP_ROOT
MODELS="${MODELS:-}"               # space-separated model suffixes; empty = auto-discover (gpt ideal excluded)
METHODS="${METHODS:-none embeddings_reranker embeddings_reranker_llm sage_time_window}"
MOD_SETS=(
  "ax" "metadata" "ocr" "screen"
  "ax metadata" "ax ocr" "ax screen"
  "metadata ocr" "metadata screen"
  "ocr screen"
  "ax metadata ocr screen"
)
IDEAL_MODEL_GLOBS=("*gpt_5.5*" "*gpt-5.5*" "*gpt*5.5*")
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
DEVICE="${DEVICE:-auto}"
NOVELTY_THRESHOLD="${NOVELTY_THRESHOLD:-0.8}"
TIME_DECAY_LAMBDA="${TIME_DECAY_LAMBDA:-5}"
TIME_WINDOW="${TIME_WINDOW:-none}"
CALIB_JSON="${RESULTS_ROOT}/soft_metric_calibration.json"
# -----------------------------------------------------------------------------

mkdir -p "$RESULTS_ROOT"

if [[ -z "$USERS" ]]; then
  USERS=$(find "$PROP_ROOT" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | tr '\n' ' ')
fi
read -r -a USER_ARR <<< "$USERS"
read -r -a METHOD_ARR <<< "$METHODS"

# ---- step 1: calibration over all users -------------------------------------
if [[ "${SKIP_CALIBRATION:-0}" == "1" && -s "$CALIB_JSON" ]]; then
  echo ">>> reusing calibration: $CALIB_JSON"
else
  echo ">>> calibrating soft metric on users: ${USER_ARR[*]}"
  calib_args=(--propositions-root "$PROP_ROOT" --users "${USER_ARR[@]}" --out "$CALIB_JSON")
  if [[ -n "$MODELS" ]]; then
    read -r -a MODEL_ARR <<< "$MODELS"
    calib_args+=(--models "${MODEL_ARR[@]}")
  fi
  "$PYTHON" scripts/calibrate_soft_metric.py "${calib_args[@]}" 2>&1 | tee "${RESULTS_ROOT}/calibration.log"
fi

SOFT_LOW=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['low'])" "$CALIB_JSON")
SOFT_HIGH=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['high'])" "$CALIB_JSON")
echo ">>> soft bounds: low=${SOFT_LOW} high=${SOFT_HIGH}"

# ---- step 2: score every run with the calibrated bounds ---------------------
discover_models() {  # $1 = user data dir; prints model suffixes, ideal excluded
  "$PYTHON" - "$1" <<'PY'
import os, sys
root = sys.argv[1]
found = set()
for name in os.listdir(root):
    if not os.path.isdir(os.path.join(root, name)):
        continue
    for m in ("ax", "metadata", "ocr", "screen"):
        if name.startswith(m + "_"):
            model = name[len(m) + 1:]
            if "gpt" not in model:
                found.add(model)
print("\n".join(sorted(found)))
PY
}

for user in "${USER_ARR[@]}"; do
  echo ""
  echo "################  USER: ${user}  ################"
  if [[ -n "$MODELS" ]]; then
    read -r -a user_models <<< "$MODELS"
  else
    user_models=()
    while IFS= read -r m; do [[ -n "$m" ]] && user_models+=("$m"); done < <(discover_models "${PROP_ROOT}/${user}")
  fi
  [[ ${#user_models[@]} -eq 0 ]] && { echo "  no models for ${user}, skipping"; continue; }

  for model in "${user_models[@]}"; do
    out_dir="${RESULTS_ROOT}/${user}/${model}"
    mkdir -p "$out_dir"
    for method in "${METHOD_ARR[@]}"; do
      for mods in "${MOD_SETS[@]}"; do
        tag="${mods// /+}"
        out_csv="${out_dir}/${method}__${tag}.csv"
        if [[ -s "$out_csv" && "${FORCE:-0}" != "1" ]]; then
          echo "--- skip (exists): ${out_csv}"; continue
        fi
        echo "--- ${user} ${model} ${method} [${mods}]"
        # shellcheck disable=SC2086  # $mods must word-split into separate modality args
        "$PYTHON" scripts/compare_modalities_vs_ideal.py \
          --propositions-root "$PROP_ROOT" \
          --user-folders "$user" \
          --candidate-model-globs "*${model}" \
          --ideal-model-globs "${IDEAL_MODEL_GLOBS[@]}" \
          --modalities $mods \
          --merge-method "$method" \
          --llm-merge-model "$LLM_MODEL" \
          --novelty-threshold "$NOVELTY_THRESHOLD" \
          --time-decay-lambda "$TIME_DECAY_LAMBDA" \
          --time-window "$TIME_WINDOW" \
          --device "$DEVICE" \
          --skip-reranker --no-progress \
          --soft-low "$SOFT_LOW" --soft-high "$SOFT_HIGH" \
          --output "$out_csv" \
          --output-dir-name "soft_${method}_${tag}" \
          > "${out_csv%.csv}.log" 2>&1 || { echo "    FAILED, see ${out_csv%.csv}.log"; rm -f "$out_csv"; }
      done
    done
  done
done

# ---- step 3: collect ---------------------------------------------------------
echo ""
echo ">>> collecting ${RESULTS_ROOT}/comparison.csv"
"$PYTHON" scripts/build_comparison.py --results-root "$RESULTS_ROOT" || true
echo ">>> done. Tables: $PYTHON scripts/build_latex_table.py --results-root $RESULTS_ROOT --metric soft_f1"
