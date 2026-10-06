#!/usr/bin/env bash
#
# The "merge observations, then propose" arm on its own, scored with the soft metrics.
#
#   [optional] generate  main_merged.py: merge one model's own summaries across modalities on
#                        wall-clock windows, then propose with that same model
#                        -> <PROP_ROOT>/<user>/merged+<mods>_<model_slug>/propositions.jsonl
#   score                compare_modalities_vs_ideal.py --skip-reranker with the calibrated
#                        soft bounds -> <RESULTS_ROOT>/<user>/<model>/merged_observations__<mods>.csv
#   collect              build_comparison.py -> <RESULTS_ROOT>/comparison.csv
#
# Generation needs the extraction model (GPU box for SGLang models, MLX models on a Mac), so it
# only runs when GENERATE_MODEL is set. Scoring alone works anywhere the ideal folders exist.
# Calibration is reused from <RESULTS_ROOT>/soft_metric_calibration.json when present (so the
# arm is scored with the same bounds as the other arms in that results root), else computed.
#
# Usage:
#   ./scripts/run_merged_arm.sh                                  # score every merged+* folder under new_data
#   USERS="hlib" MODELS="qwen3_4b" ./scripts/run_merged_arm.sh
#   GENERATE_MODEL="Qwen/Qwen3-8B" USERS="hlib" MOD_SETS="ax metadata ocr|ax metadata ocr screen" \
#       PROP_ROOT=output_data ./scripts/run_merged_arm.sh       # GPU box: generate, then score
#   WINDOW_SECONDS=120 LONG_TERM=0 ./scripts/run_merged_arm.sh   # pass-1-only variant
#   FORCE=1 ./scripts/run_merged_arm.sh                          # rescore existing CSVs
#
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- CUDA env (RTX 4090 / compute_89 requires CUDA 13 nvcc) -----------------
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_DISABLE_CUDNN_CHECK=1
# -----------------------------------------------------------------------------

# ---- config -----------------------------------------------------------------
PYTHON="${PYTHON:-.venv/bin/python}"
PROP_ROOT="${PROP_ROOT:-new_data}"
RESULTS_ROOT="${RESULTS_ROOT:-results_soft}"
USERS="${USERS:-}"                       # space-separated; empty = every folder under PROP_ROOT
MODELS="${MODELS:-}"                     # space-separated model slugs to score; empty = all of this arm's folders
GENERATE_MODEL="${GENERATE_MODEL:-}"     # full model name (e.g. Qwen/Qwen3-8B); empty = score only
SOURCE_SLUG="${SOURCE_SLUG:-}"           # slug of the folders to read when it differs from the model's slug
MOD_SETS="${MOD_SETS:-ax metadata ocr}"  # '|'-separated modality sets for generation
WINDOW_SECONDS="${WINDOW_SECONDS:-300}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.75}"
LONG_TERM="${LONG_TERM:-1}"              # 0 = stop after pass 1 (windowed de-duplication only)
MERGE_ON="${MERGE_ON:-summary-only}"     # summary-only | summary | transcription | both
declare -A FOLDER_PREFIX=([summary-only]=merged [summary]=merged-keep [transcription]=merged-tr [both]=merged-both)
PREFIX="${FOLDER_PREFIX[$MERGE_ON]}"     # only this arm's <PREFIX>+<mods>_<slug> folders are scored
K_ITEMS="${K_ITEMS:-5}"
IDEAL_MODEL_GLOBS=("*gpt_5.5*" "*gpt-5.5*" "*gpt*5.5*")
DEVICE="${DEVICE:-auto}"
CALIB_JSON="${RESULTS_ROOT}/soft_metric_calibration.json"
# -----------------------------------------------------------------------------

mkdir -p "$RESULTS_ROOT"
if [[ -z "$USERS" ]]; then
  USERS=$(find "$PROP_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '*_input_frames' -exec basename {} \; | sort | tr '\n' ' ')
fi
read -r -a USER_ARR <<< "$USERS"

# ---- generate (optional) -----------------------------------------------------
if [[ -n "$GENERATE_MODEL" ]]; then
  IFS='|' read -r -a SET_ARR <<< "$MOD_SETS"
  # One process for every user x set: the model is loaded once; finished folders are skipped.
  gen_args=(--user-names "${USER_ARR[@]}" --mod-sets "${SET_ARR[@]}"
            --model-name "$GENERATE_MODEL" --input-dir "$PROP_ROOT" --output-dir "$PROP_ROOT"
            --window-seconds "$WINDOW_SECONDS" --sim-threshold "$SIM_THRESHOLD" --k-items-per-batch "$K_ITEMS"
            --merge-on "$MERGE_ON")
  [[ -n "$SOURCE_SLUG" ]] && gen_args+=(--source-slug "$SOURCE_SLUG")
  [[ "$LONG_TERM" == "0" ]] && gen_args+=(--no-long-term)
  echo ">>> generating: ${#USER_ARR[@]} user(s) x ${#SET_ARR[@]} set(s) with ${GENERATE_MODEL}"
  "$PYTHON" -u main_merged.py "${gen_args[@]}" 2>&1 | tee "${RESULTS_ROOT}/merged_${GENERATE_MODEL//\//_}.log" \
    || echo ">>> some generation jobs failed; see ${RESULTS_ROOT}/merged_${GENERATE_MODEL//\//_}.log"
fi

# ---- calibration ---------------------------------------------------------------
if [[ -s "$CALIB_JSON" ]]; then
  echo ">>> reusing calibration: $CALIB_JSON"
else
  echo ">>> calibrating soft metric on users: ${USER_ARR[*]}"
  "$PYTHON" scripts/calibrate_soft_metric.py --propositions-root "$PROP_ROOT" --users "${USER_ARR[@]}" \
    --out "$CALIB_JSON" 2>&1 | tee "${RESULTS_ROOT}/calibration.log"
fi
SOFT_LOW=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['low'])" "$CALIB_JSON")
SOFT_HIGH=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['high'])" "$CALIB_JSON")
echo ">>> soft bounds: low=${SOFT_LOW} high=${SOFT_HIGH}"

# ---- score this arm's <PREFIX>+ folders ----------------------------------------
n_scored=0
for user in "${USER_ARR[@]}"; do
  while IFS= read -r folder; do
    name=$(basename "$folder")
    mods_tag="${name%%_*}"            # merged+ax+metadata+ocr  (the folder's modality token)
    model="${name#*_}"                # qwen3_8b
    if [[ -n "$MODELS" && " $MODELS " != *" $model "* ]]; then continue; fi
    [[ -s "$folder/propositions.jsonl" || -s "$folder/propositions.csv" ]] || { echo "--- skip (no propositions): $name"; continue; }
    out_dir="${RESULTS_ROOT}/${user}/${model}"
    mkdir -p "$out_dir"
    out_csv="${out_dir}/merged_observations__${mods_tag#"${PREFIX}+"}.csv"
    if [[ -s "$out_csv" && "${FORCE:-0}" != "1" ]]; then
      echo "--- skip (exists): ${out_csv}"; continue
    fi
    echo "--- ${user} ${model} merged_observations [${mods_tag#"${PREFIX}+"}]"
    "$PYTHON" scripts/compare_modalities_vs_ideal.py \
      --propositions-root "$PROP_ROOT" \
      --user-folders "$user" \
      --candidate-model-globs "*${model}" \
      --ideal-model-globs "${IDEAL_MODEL_GLOBS[@]}" \
      --modalities "$mods_tag" \
      --merge-method none \
      --device "$DEVICE" \
      --skip-reranker --no-progress \
      --soft-low "$SOFT_LOW" --soft-high "$SOFT_HIGH" \
      --output "$out_csv" \
      --output-dir-name "soft_merged_${mods_tag#"${PREFIX}+"}" \
      > "${out_csv%.csv}.log" 2>&1 || { echo "    FAILED, see ${out_csv%.csv}.log"; rm -f "$out_csv"; continue; }
    n_scored=$((n_scored + 1))
    "$PYTHON" - "$out_csv" <<'PY'
import csv, sys
r = next(csv.DictReader(open(sys.argv[1])))
if r.get("status") == "ok":
    print(f"    props={float(r['eval_count']):.0f}  softP={float(r['soft_precision']):.3f}  softR={float(r['soft_recall']):.3f}"
          f"  softF1={float(r['soft_f1']):.3f}  softJac={float(r['soft_jaccard']):.3f}  regions={float(r['soft_source_regions']):.0f}")
else:
    print(f"    status={r.get('status')}")
PY
  done < <(find "${PROP_ROOT}/${user}" -mindepth 1 -maxdepth 1 -type d -name "${PREFIX}+*" | sort)
done

echo ""
echo ">>> scored ${n_scored} merged run(s); collecting ${RESULTS_ROOT}/comparison.csv"
"$PYTHON" scripts/build_comparison.py --results-root "$RESULTS_ROOT" || true
