import atexit
import json
import time
from pathlib import Path
from typing import Any, List

import torch

from .base_model import BaseVisionModel
from src.configs.general_config import SGLANG_GENERATION_CONFIG, THINKING_MODELS_CONFIG, THINKING_PROMPT_SUFFIX
from src.data_models import GenerationStats, Observation, PropositionItem, PropositionSchema
from transformers import AutoProcessor


QWEN3_TEXT_MODELS = {
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-FP8",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-FP8",
}


class SGLangVisionModel(BaseVisionModel):
    def __init__(self, model_identifier: str, allowed_local_media_path: str | None = None):
        self.last_stats = None
        self.allowed_local_media_path = allowed_local_media_path or str(Path.cwd())
        self.processor = AutoProcessor.from_pretrained(model_identifier)

        self.config = SGLANG_GENERATION_CONFIG
        if model_identifier == "HuggingFaceTB/SmolVLM2-2.2B-Instruct":
            self.config["max_model_len"] = 8192

        # Mamba/hybrid models need more static memory for the state cache.
        if model_identifier in (
            "LiquidAI/LFM2.5-8B-A1B",
            "Rifky/LFM2.5-8B-A1B-FP8",
            "Qwen/Qwen3.5-9B",
            "RedHatAI/Qwen3.5-9B-FP8-dynamic",
        ):
            self.config["gpu_memory_utilization"] = 0.85

        super().__init__(model_identifier)

    def _load_model(self):
        if not torch.cuda.is_available():
            raise RuntimeError("SGLangVisionModel requires a CUDA GPU, but torch.cuda.is_available() is False.")

        try:
            import sglang as sgl
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import SGLang. Recreate the CUDA environment with: "
                "uv sync --extra cuda --reinstall"
            ) from exc

        print(f"Loading SGLang model: {self.model_identifier}...")
        self.model = sgl.Engine(
            model_path=self.model_identifier,
            tp_size=self.config["tensor_parallel_size"],
            dp_size=self.config["data_parallel_size"],
            context_length=self.config["max_model_len"],
            mem_fraction_static=self.config["gpu_memory_utilization"],
            limit_mm_data_per_request=self.config["limit_mm_data_per_request"],
        )
        self._close_callback = self.close
        atexit.register(self._close_callback)
        print("SGLang model loaded successfully!")

    def get_response_stats(self) -> GenerationStats | None:
        return self.last_stats

    def close(self) -> None:
        close_callback = getattr(self, "_close_callback", None)
        if close_callback is not None:
            atexit.unregister(close_callback)
            self._close_callback = None

        model = getattr(self, "model", None)
        if model is None:
            return

        self.model = None
        model.shutdown()

    def _build_prompt(self, prompt: str, image_count: int = 0) -> str:
        is_qwen3_text_model = self.model_identifier in QWEN3_TEXT_MODELS

        if is_qwen3_text_model:
            content = prompt
        else:
            content = [{"type": "image"} for _ in range(image_count)]
            content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        is_thinking_prompt_model = self.model_identifier in THINKING_MODELS_CONFIG["thinking-prompt"]

        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False if is_thinking_prompt_model else self.config["enable_thinking"],
        )

        if is_thinking_prompt_model:
            rendered += THINKING_PROMPT_SUFFIX[self.model_identifier]

        return rendered

    @staticmethod
    def _extract_output_text(output: Any) -> str:
        if isinstance(output, list):
            output = output[0]
        if isinstance(output, dict):
            if "text" in output:
                return output["text"]
            if output.get("outputs"):
                return output["outputs"][0].get("text", "")
        return getattr(output, "text", str(output))

    @staticmethod
    def _extract_token_count(output: Any, *keys: str) -> int:
        if isinstance(output, list):
            output = output[0]
        if isinstance(output, dict):
            meta_info = output.get("meta_info") or {}
        else:
            meta_info = getattr(output, "meta_info", {}) or {}
        for key in keys:
            value = meta_info.get(key)
            if value is not None:
                return int(value)
        return 0

    def _run_inference(
        self,
        prompt: str,
        images: List[str] | None,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        if images is None:
            images = []

        image_paths = [str(Path(image).expanduser().resolve()) for image in images]
        sampling_params = {
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
            "repetition_penalty": self.config["repetition_penalty"],
            "max_new_tokens": max_tokens,
        }
        if json_schema is not None:
            # SGLang's offline Engine API expects the schema as a JSON string.
            # Constrained decoding prevents prose, fences, and malformed JSON.
            sampling_params["temperature"] = 0.0
            sampling_params["json_schema"] = json.dumps(json_schema)

        start = time.perf_counter()
        output = self.model.generate(
            prompt=self._build_prompt(prompt, image_count=len(image_paths)),
            image_data=image_paths if image_paths else None,
            sampling_params=sampling_params,
        )
        latency = time.perf_counter() - start

        output_text = self._extract_output_text(output)
        prompt_tokens = self._extract_token_count(output, "prompt_tokens", "input_tokens")
        generation_tokens = self._extract_token_count(output, "completion_tokens", "output_tokens")
        self.last_stats = GenerationStats(
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            total_tokens=prompt_tokens + generation_tokens,
            latency_sec=latency,
            prompt_tps=prompt_tokens / latency if latency > 0 else 0.0,
            generation_tps=generation_tokens / latency if latency > 0 else 0.0,
            peak_memory=-1,
        )
        return output_text

    def generate_response(self, images: List[str] | None = None, prompt: str = "", verbose: bool = False) -> str:
        output_text = self._run_inference(
            prompt=prompt,
            images=images,
            max_tokens=self.config["max_tokens"],
        )
        if verbose:
            print(f"Generated: {output_text}")
        return output_text

    def generate_propositions(self, observation: Observation, propose_prompt: str, user_name: str) -> list[PropositionItem]:
        prompt = (
            propose_prompt.replace("{user_name}", user_name)
            .replace("{inputs}", observation.content)
        )
        output_text = self._run_inference(
            prompt=prompt,
            images=None,
            max_tokens=self.config["max_tokens_propositions"],
            json_schema=PropositionSchema.model_json_schema(),
        )
        validated = PropositionSchema.model_validate_json(output_text)
        return validated.model_dump()

    def generate_response_batch(
        self,
        prompts: List[str],
        images_list: List[List[str]] | None = None,
        max_tokens: int | None = None,
    ) -> list[tuple[str, "GenerationStats"]]:
        """Generate free-text responses for many prompts in one batched call.

        Optionally multimodal: ``images_list[i]`` are the image paths for
        ``prompts[i]`` (omit or leave empty for text-only prompts). Used to fan
        the provider transcription/summary stages across all data-parallel
        replicas instead of one GPU at a time. Returns (text, stats) per prompt.
        """
        return self._run_inference_batch(
            prompts=prompts,
            max_tokens=max_tokens if max_tokens is not None else self.config["max_tokens"],
            images_list=images_list,
        )

    def _run_inference_batch(
        self,
        prompts: List[str],
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
        images_list: List[List[str]] | None = None,
    ) -> list[tuple[str, GenerationStats]]:
        """Submit many prompts in a single Engine.generate() call.

        A single request only ever lands on one data-parallel replica, so
        submitting prompts one-at-a-time leaves the other GPUs idle. Passing a
        list lets SGLang's scheduler spread the requests across all dp_size
        replicas. ``images_list`` (one image-path list per prompt) enables
        multimodal batches; leave it None for text-only.
        """
        sampling_params = {
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
            "repetition_penalty": self.config["repetition_penalty"],
            "max_new_tokens": max_tokens,
        }
        if json_schema is not None:
            sampling_params["temperature"] = 0.0
            sampling_params["json_schema"] = json.dumps(json_schema)

        if images_list is None:
            images_list = [[] for _ in prompts]
        resolved_images = [
            [str(Path(image).expanduser().resolve()) for image in images]
            for images in images_list
        ]
        built_prompts = [
            self._build_prompt(prompt, image_count=len(images))
            for prompt, images in zip(prompts, resolved_images)
        ]
        # Per-request image_data (None where a prompt has no images). Omit
        # entirely when the whole batch is text-only.
        image_data = None
        if any(resolved_images):
            image_data = [images or None for images in resolved_images]

        start = time.perf_counter()
        # A single sampling_params dict is broadcast to every prompt in the batch.
        outputs = self.model.generate(
            prompt=built_prompts, image_data=image_data, sampling_params=sampling_params
        )
        latency = time.perf_counter() - start

        if not isinstance(outputs, list):
            outputs = [outputs]

        # The offline Engine API does not expose per-request timing, so every
        # item in the chunk reports the shared wall-clock latency.
        results: list[tuple[str, GenerationStats]] = []
        for out in outputs:
            text = self._extract_output_text(out)
            prompt_tokens = self._extract_token_count(out, "prompt_tokens", "input_tokens")
            generation_tokens = self._extract_token_count(out, "completion_tokens", "output_tokens")
            stats = GenerationStats(
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                total_tokens=prompt_tokens + generation_tokens,
                latency_sec=latency,
                prompt_tps=prompt_tokens / latency if latency > 0 else 0.0,
                generation_tps=generation_tokens / latency if latency > 0 else 0.0,
                peak_memory=-1,
            )
            results.append((text, stats))
        return results

    def generate_propositions_batch(
        self, observations: List[Observation], propose_prompt: str, user_name: str
    ) -> list[dict[str, Any]]:
        """Generate propositions for many observations in one batched call.

        Returns one result dict per observation, in the same order:
          {"propositions": [...], "stats": GenerationStats}      on success
          {"error": "<Type>: <msg>", "stats": GenerationStats}   if the model
        output fails schema validation. Parse failures are isolated per item so
        one bad output does not fail the whole chunk.
        """
        prompts = [
            propose_prompt.replace("{user_name}", user_name).replace("{inputs}", obs.content)
            for obs in observations
        ]
        raw = self._run_inference_batch(
            prompts=prompts,
            max_tokens=self.config["max_tokens_propositions"],
            json_schema=PropositionSchema.model_json_schema(),
        )
        results: list[dict[str, Any]] = []
        for text, stats in raw:
            try:
                validated = PropositionSchema.model_validate_json(text)
                results.append({"propositions": validated.model_dump()["propositions"], "stats": stats})
            except Exception as e:  # noqa: BLE001 - isolate per-item parse failures
                results.append({"error": f"{type(e).__name__}: {e}"[:500], "stats": stats})
        return results
