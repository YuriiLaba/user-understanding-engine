import time
from typing import List, Optional, Any
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from .base_model import BaseVisionModel
from src.data_models import Observation, PropositionItem, GenerationStats
import json
from src.configs.general_config import MLX_GENERATION_CONFIG
from src.helpers.json_parsing import parse_model_json
import re

def remove_think(text: str) -> str:
    # Removes <think>...</think> including the tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


class MLXModel(BaseVisionModel):
    def __init__(self, model_identifier: str, *args, **kwargs):
        super().__init__(model_identifier, *args, **kwargs)
        self.last_stats: Optional[GenerationStats] = None
    
    def _load_model(self):
        self.model, self.tokenizer = load(self.model_identifier)
        print(f"{self.model_identifier} model loaded successfully!")
    
    def _run_inference(self, input_prompt: Any, max_tokens) -> str:
        
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": input_prompt}]
            # Same protocol as the SGLang path (SGLANG_GENERATION_CONFIG["enable_thinking"] = False):
            # Qwen3-style templates take the flag; templates without it ignore the extra kwarg.
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False)
        
        start = time.perf_counter()

        # mlx_lm >= 0.20: generate() returns only text; stream_generate() carries the stats.
        sampler = make_sampler(temp=MLX_GENERATION_CONFIG["temperature"], top_p=MLX_GENERATION_CONFIG["top_p"])
        output, last = "", None
        for last in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
            output += last.text

        latency = time.perf_counter() - start

        prompt_tokens = getattr(last, "prompt_tokens", 0) if last else 0
        generation_tokens = getattr(last, "generation_tokens", 0) if last else 0
        self.last_stats = GenerationStats(
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            total_tokens=prompt_tokens + generation_tokens,
            latency_sec=latency,
            prompt_tps=getattr(last, "prompt_tps", 0.0) if last else 0.0,
            generation_tps=getattr(last, "generation_tps", 0.0) if last else 0.0,
            peak_memory=getattr(last, "peak_memory", 0.0) if last else 0.0,
        )
        return remove_think(output)
    
    def get_response_stats(self) -> Optional[GenerationStats]:
        """Get statistics about the last generation.

        Returns:
            GenerationStats of generation statistics
        """
        if self.last_stats is None:
            return None
        return self.last_stats

    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response given a text prompt.

        Args:
            prompt: User prompt text (plain).

        Returns:
            Generated text (string).
        """

        return self._run_inference(input_prompt=prompt, max_tokens=MLX_GENERATION_CONFIG["max_tokens"])
    
    def generate_propositions(self, observation: Observation, propose_prompt: str, user_name: str) -> list[PropositionItem]:
        """Generate propositions from an observation.
        
        Args:
            observation: The observation to generate propositions from.
            propose_prompt: The prompt template for proposing
            user_name: Name of the user for personalization
            
        Returns:
            List of generated propositions.
        """
        prompt = (
            propose_prompt.replace("{user_name}", user_name)
            .replace("{inputs}", observation.content)
        )

        text = self._run_inference(input_prompt=prompt, max_tokens=MLX_GENERATION_CONFIG["max_tokens_propositions"])
        return parse_model_json(text)