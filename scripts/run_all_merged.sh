#!/usr/bin/env bash
# Run the merged-observations arm for every model across all applicable modality sets.
#
# Usage:
#   ./scripts/run_all_merged.sh
#   PROP_ROOT=output_data RESULTS_ROOT=results_soft_merged_observation ./scripts/run_all_merged.sh
#   USERS="hlib anastasia" ./scripts/run_all_merged.sh
#   DRY_RUN=1 ./scripts/run_all_merged.sh   # print commands without running
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

PROP_ROOT="${PROP_ROOT:-output_data}"
RESULTS_ROOT="${RESULTS_ROOT:-results_soft_merged_observation}"
USERS="${USERS:-}"
DRY_RUN="${DRY_RUN:-0}"
MERGE_ON="${MERGE_ON:-summary-only}"     # summary-only | summary | transcription | both

# ---- full modality sets to attempt (space-separated within each set) --------
ALL_MOD_SETS=(
  "ax"
  "metadata"
  "ocr"
  "screen"
  "ax metadata"
  "ax ocr"
  "ax screen"
  "metadata ocr"
  "metadata screen"
  "ocr screen"
  "ax metadata ocr screen"
)

# ---- slug -> full model name ------------------------------------------------
declare -A MODEL_NAME=(
  [qwen3_4b]="Qwen/Qwen3-4B"
  [qwen3_8b]="Qwen/Qwen3-8B"
  [qwen3.5_4b]="Qwen/Qwen3.5-4B"
  [qwen3.5_9b]="Qwen/Qwen3.5-9B"
  [gemma_4_e4b_it]="google/gemma-4-E4B-it"
  [granite_4.1_3b]="ibm-granite/granite-4.1-3b"
  [granite_4.1_8b]="ibm-granite/granite-4.1-8b"
  [lfm2.5_8b_a1b]="LiquidAI/LFM2.5-8B-A1B"
  [lfm2.5_vl_1.6b]="LiquidAI/LFM2.5-VL-1.6B"
  [nvidia_nemotron_3_nano_4b_bf16]="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
  [qwen3_vl_4b_instruct]="Qwen/Qwen3-VL-4B-Instruct"
  [qwen3_vl_8b_instruct]="Qwen/Qwen3-VL-8B-Instruct"
)

# Discover users if not specified
if [[ -z "$USERS" ]]; then
  USERS=$(find "$PROP_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '*_input_frames' -exec basename {} \; | sort | tr '\n' ' ')
fi
read -r -a USER_ARR <<< "$USERS"

# ---- helper: check all modalities in a set exist for a user+slug -----------
mods_available() {
  local user="$1" slug="$2" mods="$3"
  for m in $mods; do
    [[ -d "${PROP_ROOT}/${user}/${m}_${slug}" ]] || return 1
  done
  return 0
}

# ---- main loop --------------------------------------------------------------
slugs=("${!MODEL_NAME[@]}")
n_models=${#slugs[@]}
model_idx=0

for slug in "${slugs[@]}"; do
  model_idx=$((model_idx + 1))
  full_model="${MODEL_NAME[$slug]}"

  # Build the pipe-separated MOD_SETS string, keeping only applicable sets.
  # A set is applicable if ALL its modalities exist for at least one user.
  applicable=()
  for mods in "${ALL_MOD_SETS[@]}"; do
    for user in "${USER_ARR[@]}"; do
      if mods_available "$user" "$slug" "$mods"; then
        applicable+=("$mods")
        break
      fi
    done
  done

  if [[ ${#applicable[@]} -eq 0 ]]; then
    echo ">>> [${model_idx}/${n_models}] skip $slug — no applicable mod sets found"
    continue
  fi

  # Join with |
  pipe_sets=$(printf "%s|" "${applicable[@]}")
  pipe_sets="${pipe_sets%|}"  # strip trailing |

  echo ""
  echo ">>> [${model_idx}/${n_models}] model: $slug"
  echo "    sets (${#applicable[@]}): $(IFS='|'; echo "${applicable[*]}")"

  cmd=(env
    PROP_ROOT="$PROP_ROOT"
    RESULTS_ROOT="$RESULTS_ROOT"
    GENERATE_MODEL="$full_model"
    USERS="$USERS"
    MOD_SETS="$pipe_sets"
    MERGE_ON="$MERGE_ON"
    ./scripts/run_merged_arm.sh
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    DRY_RUN: ${cmd[*]}"
  else
    t_start=$(date +%s)
    "${cmd[@]}"
    t_end=$(date +%s)
    echo ">>> [${model_idx}/${n_models}] $slug done in $((t_end - t_start))s"
  fi
done

echo ""
echo ">>> All ${n_models} models processed."
