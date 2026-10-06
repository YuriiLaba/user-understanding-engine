import time
from typing import List, Optional, Any
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from .base_model import BaseVisionModel
from src.data_models import Observation, PropositionItem, GenerationStats
import json
from src.configs.general_config import MLX_GENERATION_CONFIG
from src.helpers.json_parsing import parse_model_json


import re
import json

def extract_json(text: str):
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if not json_match:
        return

    json_str = json_match.group(1)

    json_str = json_str.replace("\n", "").strip()

    return json_str

def remove_repeated_ending(text):
    parts = text.split()
    # Walk backward while the last token repeats
    while len(parts) > 1 and parts[-1] == parts[-2]:
        parts.pop()
    return " ".join(parts)



class MLXVisionModel(BaseVisionModel):
    def __init__(self, model_identifier: str, *args, **kwargs):
        super().__init__(model_identifier, *args, **kwargs)
        self.last_stats: Optional[GenerationStats] = None
    
    def _load_model(self):
        self.model, self.processor = load(self.model_identifier)
        self.config = self.model.config
        print(f"{self.model_identifier} model loaded successfully!")
    
    def _run_inference(self, input_prompt: Any, images: Optional[List[str]] = None, max_tokens=None) -> str:
        
        if images is None:
            print("IMAGES IS NONE!")
            images = []

        start = time.perf_counter()
        output = generate(
            self.model,
            self.processor,
            input_prompt,
            images,
            verbose=MLX_GENERATION_CONFIG["verbose"],
            temperature=MLX_GENERATION_CONFIG["temperature"],
            top_p=MLX_GENERATION_CONFIG["top_p"],
            max_tokens=max_tokens,
            image_aspect_ratio=MLX_GENERATION_CONFIG["image_aspect_ratio"]            
        )
        latency = time.perf_counter() - start

        self.last_stats = GenerationStats(
            prompt_tokens=getattr(output, "prompt_tokens", 0),
            generation_tokens=getattr(output, "generation_tokens", 0),
            total_tokens=getattr(output, "total_tokens", 0),
            latency_sec=latency,
            prompt_tps=getattr(output, "prompt_tps", 0.0), # tokens per second
            generation_tps=getattr(output, "generation_tps", 0.0),
            peak_memory=getattr(output, "peak_memory", 0.0), # the maximum GPU/Metal memory (VRAM) the model used during the generation call
        )
        return output.text
    
    def get_response_stats(self) -> Optional[GenerationStats]:
        """Get statistics about the last generation.

        Returns:
            GenerationStats of generation statistics
        """
        if self.last_stats is None:
            return None
        return self.last_stats

    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response given images + a text prompt.

        Args:
            prompt: User prompt text (plain).

        Returns:
            Generated text (string).
        """
        images = kwargs.get("images")
        if images is None:
            images = []

        processed_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=len(images)
        )

        return self._run_inference(input_prompt=processed_prompt, images=images, max_tokens=MLX_GENERATION_CONFIG["max_tokens"])
    
    def generate_propositions(self, observation: Observation, propose_prompt: str, user_name: str) -> list[PropositionItem]:
        """Generate propositions from an observation.
        
        Args:
            observation: The observation to generate propositions from.
            propose_prompt: The prompt template for proposing
            user_name: Name of the user for personalization
            
        Returns:
            List of generated propositions.
        """

        import re

        clean = re.sub(r"[^A-Za-z]+", " ", observation.content)
        clean = clean.strip()
        clean = remove_repeated_ending(clean)

        prompt = (
            propose_prompt.replace("{user_name}", user_name)
            .replace("{inputs}", clean)
        )
        
        processed_prompt = apply_chat_template(
            self.processor, self.config, prompt
        )


        text = self._run_inference(input_prompt=processed_prompt, max_tokens=MLX_GENERATION_CONFIG["max_tokens_propositions"])
        return parse_model_json(text)