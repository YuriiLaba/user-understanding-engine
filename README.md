# user-understanding-engine

Infers a structured profile of a computer user from raw observations of their activity.
Three input modalities feed the same pipeline: screen recordings, the macOS accessibility
tree (AX), and system metadata (app/mouse/keyboard events).

## Build

Dependencies are managed with **uv**. The `cuda` (Linux/SGLang) and `macos` (Apple Silicon/MLX)
extras are mutually exclusive — install only one.

### CUDA / SGLang (Linux)

```bash
uv python install 3.12
uv sync --python 3.12 --extra cuda --reinstall
VIRTUAL_ENV=.venv uv pip install --no-deps --reinstall nvidia-cudnn-cu12==9.16.0.29
```

Use Python 3.12 (some SGLang deps build from source on 3.13). The CuDNN reinstall overrides the
older wheel pinned by PyTorch, which SGLang rejects at runtime.

The 4090/Ada GPUs (arch 8.9) need a recent CUDA toolkit. If you see
`Unsupported gpu architecture 'compute_89'`, point the shell at CUDA 13.0 **before** building/running:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.9"
hash -r && nvcc --version   # must report 13.0, not the default 11.5
```

If an old CUDA compiler was used before, clear stale kernel caches:
`rm -rf ~/.cache/{torch_extensions,triton,sglang,flashinfer}`.

### macOS / MLX

```bash
uv sync --extra macos --extra openai
```

### Other requirements

- `OPENAI_API_KEY` in `.env` — the final profile stage always calls OpenAI `gpt-4o`, regardless of the local model.
- For screen runs with `--apply-ocr` on Linux, install Tesseract with English + Ukrainian data:
  `sudo apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-ukr`. Pure vision (`--no-apply-ocr`) needs no OCR.

## Run

The three modality entry points are driven entirely by CLI args. `--user-name`, `--model-name`,
and the modality's input path are **required**; the model must be listed in `SUPPORTED_MODELS`
(`src/configs/general_config.py`). On the CUDA box, prefix runs with the SGLang env vars.

```bash
SGLANG_DISABLE_CUDNN_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2 \
uv run --no-sync python main_screen.py \
  --user-name hlib --model-name Qwen/Qwen3-VL-4B-Instruct --no-apply-ocr \
  --paths-to-videos raw_data/hlib/video1.mp4 raw_data/hlib/video2.mp4

uv run --no-sync python main_ax.py \
  --user-name hlib --model-name Qwen/Qwen3-VL-4B-Instruct \
  --path-to-accessibility raw_data/hlib/accessibility.json

uv run --no-sync python main_metadata.py \
  --user-name hlib --model-name Qwen/Qwen3-VL-4B-Instruct \
  --path-to-metadata raw_data/hlib   # dir holding app/mouse/keyboard_events.json
```

Pass `--help` to any entry point for the full flag list. Outputs land in
`output_data/{user_name}/{experiment_prefix}_{model}/` (transcriptions, summaries, propositions,
`user_profile.json`, and a `params.json` snapshot of the run config). **A run aborts if its output
dir already exists** — delete it or change `--user-name` / `--experiment-prefix` to re-run.

### Batch runs

`run_experiments.sh` runs every modality for each model in its `MODELS` array, exporting the CUDA env
once and writing per-run logs to `logs/`:

```bash
./run_experiments.sh
```

## Evaluation

Profiles are scored against a reference built from the `gpt_5.5` runs of the same sessions, with
soft precision / recall / F1 / Jaccard. All scripts take `PROP_ROOT` (default `new_data`; use
`output_data` here) and skip results that already exist (`FORCE=1` recomputes).

```bash
# 1. Proposition-level arms (none, SAGE, rerank, rerank+LLM): calibrate once, then score
PROP_ROOT=output_data RESULTS_ROOT=results_soft ./scripts/run_soft_metrics.sh
#    add SKIP_CALIBRATION=1 to reuse results_soft/soft_metric_calibration.json

# 2. Observation-level arms: merge each model's summaries across modalities, re-propose, score
#    (GPU; MERGE_ON picks the arm and its folder prefix)
MERGE_ON=summary-only RESULTS_ROOT=results_soft_merged_observation ./scripts/run_all_merged.sh  # merged+
MERGE_ON=summary      RESULTS_ROOT=results_soft_merged_summary     ./scripts/run_all_merged.sh  # merged-keep+

# 3. Paper table (merging method x modality subset) from all three roots
python3 scripts/build_latex_table.py --metric soft_f1   # -> results_soft/table_merging_soft_f1.tex
```

Copy the calibration JSON into each results root so every arm is scored with the same bounds.
Each root's `comparison.csv` collects its runs. The table pairs runs across columns and reports
the 95% CI over models. A text-only model's all-four run counts as AX+Meta+OCR.

If an observation-level run is interrupted, a partially generated folder (no `propositions.csv`)
may already have been scored. Delete that CSV before resuming, or the resumed run keeps it.

## Troubleshooting

### `resource_tracker: process died unexpectedly` / `KeyError: '/loky-...'` at shutdown

This traceback appears **after** a run finishes and is **harmless** — outputs are already written.
It is not raised by our code. SGLang's deep dependency chain transitively pulls in `joblib`, whose
`loky` process-pool backend ships its **own** resource tracker alongside CPython's
`multiprocessing.resource_tracker`. At interpreter shutdown the two race to unlink the same
`/loky-*` POSIX semaphore: one cleans it, the other then calls `cache.remove(name)` on a name that's
already gone → `KeyError`. The accompanying `process died unexpectedly` happens because SGLang tears
down its worker process group hard, killing the tracker child mid-cleanup. Worst case is a leftover
semaphore in `/dev/shm`, reclaimed on reboot. Safe to ignore; upgrading `joblib`/`loky` may quiet it.

