#!/usr/bin/env bash
#
# Run the 4 baseline merge arms against the ideal (all-modality gpt_5.5 union),
# for every user. For each candidate model, each single modality AND all four
# together are run.
#
# Metrics per run: embedding-intersection + reranker classification (precision/
# recall/f1). Summary CSVs land in results/<user>/<model>/, reranker coverage/
# precision JSONL land inside the user's data folder.
#
# Usage:
#   ./scripts/run_baselines.sh                 # run everything
#   ./scripts/run_baselines.sh 2>&1 | tee run.log
#
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- config -----------------------------------------------------------------
PYTHON=".venv/bin/python"
PROP_ROOT="new_data"
USERS=("anastasia" "anastasia_video" "hlib")

# Candidate models to evaluate (each becomes a glob "*<model>").
# Leave empty to auto-discover ALL models per user except the gpt_5.5 ideal.
# To pin a subset, list them, e.g. MODELS=("qwen3_vl_4b_instruct" "granite_4.1_8b").
MODELS=()

# The 4 baseline merge arms.
METHODS=("none" "embeddings_reranker" "embeddings_reranker_llm" "sage_time_window")

# Modality sets: each modality alone, every pair, then all four together.
MOD_SETS=(
  "ax" "metadata" "ocr" "screen"
  "ax metadata" "ax ocr" "ax screen"
  "metadata ocr" "metadata screen"
  "ocr screen"
  "ax metadata ocr screen"
)

# Ideal = all four modalities of gpt_5.5.
IDEAL_MODEL_GLOBS=("*gpt_5.5*" "*gpt-5.5*" "*gpt*5.5*")

BATCH_SIZE=256   # reranker throughput peaks here on MPS (benchmarked)
LLM_MODEL="gpt-4o-mini"
RESULTS_ROOT_BASE="results"
# -----------------------------------------------------------------------------

discover_models() {  # $1 = data dir for a user; prints model names (gpt ideal excluded)
  "$PYTHON" - "$1" <<'PY'
import os, sys
root = sys.argv[1]
modalities = ("ax", "metadata", "ocr", "screen")
found = set()
for name in os.listdir(root):
    if not os.path.isdir(os.path.join(root, name)):
        continue
    for m in modalities:
        if name.startswith(m + "_"):
            model = name[len(m) + 1:]
            if "gpt" not in model:  # exclude the ideal model
                found.add(model)
for model in sorted(found):
    print(model)
PY
}

for user in "${USERS[@]}"; do
  echo ""
  echo "################################################################"
  echo " USER: ${user}"
  echo "################################################################"
  results_root="${RESULTS_ROOT_BASE}/${user}"
  mkdir -p "$results_root"

  # Models for this user: pinned MODELS, else auto-discovered from the user's data.
  if [[ ${#MODELS[@]} -gt 0 ]]; then
    user_models=("${MODELS[@]}")
  else
    user_models=()
    while IFS= read -r m; do
      [[ -n "$m" ]] && user_models+=("$m")
    done < <(discover_models "${PROP_ROOT}/${user}")
    echo "  [${user}] discovered ${#user_models[@]} models: ${user_models[*]:-(none)}"
  fi

  if [[ ${#user_models[@]} -eq 0 ]]; then
    echo "  [${user}] no candidate models found - skipping user"
    continue
  fi

  for model in "${user_models[@]}"; do
    cand_glob="*${model}"
    out_dir="${results_root}/${model}"
    mkdir -p "$out_dir"

    for method in "${METHODS[@]}"; do
      for mods in "${MOD_SETS[@]}"; do
        tag="${mods// /+}"
        out_csv="${out_dir}/${method}__${tag}.csv"

        # Skip if we already have results for this method+modality combination.
        if [[ -s "$out_csv" ]]; then
          echo "--- SKIP (exists): ${out_csv}"
          continue
        fi

        echo ""
        echo "=================================================================="
        echo " user=${user}  model=${model}  method=${method}  modalities=[${mods}]"
        echo " -> ${out_csv}"
        echo "=================================================================="

        $PYTHON scripts/compare_modalities_vs_ideal.py \
          --propositions-root "$PROP_ROOT" \
          --user-folders "$user" \
          --candidate-model-globs "$cand_glob" \
          --ideal-model-globs "${IDEAL_MODEL_GLOBS[@]}" \
          --modalities $mods \
          --merge-method "$method" \
          --llm-merge-model "$LLM_MODEL" \
          --batch-size "$BATCH_SIZE" \
          --output "$out_csv" \
          --output-dir-name "reranker_${method}_${tag}"
      done
    done
  done
done

echo ""
echo "All runs complete. Summaries under ${RESULTS_ROOT_BASE}/<user>/"
