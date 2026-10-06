SUPPORTED_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-5.1", "gpt-5.5"],
    "sglang": [
        # checked models
        "Qwen/Qwen3.5-4B",
        "RedHatAI/Qwen3.5-4B-FP8-dynamic",
        "Qwen/Qwen3.5-9B",
        "RedHatAI/Qwen3.5-9B-FP8-dynamic",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B-FP8",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B-FP8",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct-FP8",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "google/gemma-4-E4B-it",
        "LiquidAI/LFM2.5-VL-1.6B",
        "LiquidAI/LFM2.5-8B-A1B",
        "Rifky/LFM2.5-8B-A1B-FP8",
        "ibm-granite/granite-4.1-8b",
        "ibm-granite/granite-4.1-8b-fp8",
        "ibm-granite/granite-4.1-3b",
        "ibm-granite/granite-4.1-3b-fp8",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
    ],
    "mlx": [
        "mlx-community/Qwen3-VL-4B-Instruct-6bit",
        "mlx-community/SmolVLM-Instruct-6bit",
        "mlx-community/gemma-3-4b-it-6bit",
        "mlx-community/gemma-3-12b-it-6bit",
        "mlx-community/gemma-3n-E4B-it-6bit",
    ],
    "mlx_lm": [
        "mlx-community/Qwen3-4B-6bit",
        "Qwen/Qwen3-4B-MLX-6bit",
        "mlx-community/granite-3.3-2b-instruct-6bit",
        "mlx-community/granite-3.3-8b-instruct-6bit",
        "mlx-community/granite-3.3-2b-instruct-8bit",
    ],
}

MLX_GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 1.0,
    "max_tokens": 1_000,
    "image_aspect_ratio": "pad",  # "pad" (default) "resize" "pad" keeps the entire image means more stable recognition.
    "verbose": False,
    "max_tokens_propositions": 12_000,
}

OPEN_AI_GENERATION_CONFIG = {
    "detail": "high",
    "temperature": 0.1,
    # "max_tokens": 100,
    # "max_tokens_propositions": 10_000,
}

# Models that only accept the default temperature (1); passing any other value raises 400.
OPENAI_FIXED_TEMPERATURE_MODELS = {"gpt-5.5", "o1", "o1-mini", "o3", "o3-mini"}

SGLANG_GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 1.0,
    "repetition_penalty": 1.1,  # >1 discourages token loops (e.g. "1.1. 1.1. ..." runaways)
    "max_tokens": 5_000,
    "max_tokens_propositions": 12_000,
    "max_model_len": 40_960, # 8192 for SmolVLM2
    "tensor_parallel_size": 1,
    "data_parallel_size": 3,
    "gpu_memory_utilization": 0.80,
    "limit_mm_data_per_request": {"image": 10},
    "enable_thinking": False, # We need it as False 
}

THINKING_MODELS_CONFIG = {
    "thinking-non": [  # models that do not have thinking mode
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct-FP8",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "LiquidAI/LFM2.5-VL-1.6B",
        "ibm-granite/granite-4.1-8b",
        "ibm-granite/granite-4.1-8b-fp8",
        "ibm-granite/granite-4.1-3b",
        "ibm-granite/granite-4.1-3b-fp8",
    ],
    "thinking-flag": [  # models that have thinking mode and it can be switched with a flag
        "Qwen/Qwen3.5-4B",
        "RedHatAI/Qwen3.5-4B-FP8-dynamic",
        "Qwen/Qwen3.5-9B",
        "RedHatAI/Qwen3.5-9B-FP8-dynamic",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B-FP8",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
    ],
    "thinking-prompt": [  # models that have thinking mode and it cannot be switched, so we prepend an empty <think></think> block / see sglang_model.py, __build_prompt
        "LiquidAI/LFM2.5-8B-A1B",
        "Rifky/LFM2.5-8B-A1B-FP8",
        # Gemma4 E4B is included here too
        "google/gemma-4-E4B-it",
    ],
}

THINKING_PROMPT_SUFFIX = {
    "LiquidAI/LFM2.5-8B-A1B": "<think>\n\n</think>\n\n",
    "Rifky/LFM2.5-8B-A1B-FP8": "<think>\n\n</think>\n\n",
    "google/gemma-4-E4B-it": "<|channel>thought\n<channel|>",
}