import time
import json
import base64
from typing import List, Optional
from openai import OpenAI
from .base_model import BaseVisionModel
from src.data_models import Observation, PropositionItem, PropositionSchema, GenerationStats
from src.helpers.json_parsing import parse_model_json
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any
from src.configs.general_config import OPEN_AI_GENERATION_CONFIG, OPENAI_FIXED_TEMPERATURE_MODELS

MEDIA_TYPE_MAP = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp'
}

class OpenAIVisionModel(BaseVisionModel):
    def __init__(self, model_identifier: str, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None):
        """
        Initialize OpenAI model
        
        Args:
            model_identifier: Model name (e.g., "gpt-4o-mini")
            api_key: Optional API key
            base_url: Optional base URL for custom endpoints
        """
        self.api_key = api_key
        self.base_url = base_url
        self.last_stats: Optional[GenerationStats] = None
        super().__init__(model_identifier)
    
    def _load_model(self):
        client_kwargs = {}
        if self.api_key:
            client_kwargs['api_key'] = self.api_key
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
        
        self.client = OpenAI(**client_kwargs)
        print(f"{self.model_identifier} model loaded successfully!")
    
    @staticmethod
    def _encode_image_to_base64(image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.standard_b64encode(f.read()).decode('utf-8')
    
    @staticmethod
    def _guess_media_type(image_path: Path) -> str:
        """Infer media type from file extension."""
        return MEDIA_TYPE_MAP.get(image_path.suffix.lower(), "image/jpeg")

    def _create_image_content_block(
        self,
        image_path: Path
    ) -> Dict[str, Any]:
        """
        Create image content block for the API.

        Args:
            image_path: Path to image.

        Returns:
            Image content dictionary.
        """
        image_data = self._encode_image_to_base64(image_path)
        media_type = self._guess_media_type(image_path)

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_data}",
                "detail": OPEN_AI_GENERATION_CONFIG["detail"],
            },
        }
    
    def _run_inference(
        self,
        messages: Sequence[Dict[str, Any]],
    ):
        """
        Thin wrapper around chat.completions.create to avoid repetition.
        """

        schema = PropositionSchema.model_json_schema()

        start = time.perf_counter()
        extra = {} if self.model_identifier in OPENAI_FIXED_TEMPERATURE_MODELS else \
            {"temperature": OPEN_AI_GENERATION_CONFIG["temperature"]}

        output = self.client.chat.completions.create(
            model=self.model_identifier,
            messages=messages,
            max_completion_tokens=12_000,
            **extra,
            timeout=60.0,
            response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "proposition_schema",
                        "strict": True,
                        "schema": schema
                    }
                }
        )

        latency_sec = time.perf_counter() - start

        usage = output.usage
        prompt_tokens = usage.prompt_tokens
        generation_tokens = usage.completion_tokens

        self.last_stats = GenerationStats(
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            total_tokens=usage.total_tokens,
            latency_sec=latency_sec,
            prompt_tps=prompt_tokens / latency_sec if latency_sec > 0 else 0.0,
            generation_tps=generation_tokens / latency_sec if latency_sec > 0 else 0.0,
            peak_memory=-1,  # OpenAI models don't expose memory stats
        )

        return output
    
    def get_response_stats(self) -> Optional[GenerationStats]:
        """Get statistics about the last generation.

        Returns:
            GenerationStats of generation statistics
        """
        if self.last_stats is None:
            return None
        return self.last_stats

    def generate_response(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate response using OpenAI vision model.

        Args:
            prompt: Text prompt.

        Returns:
            Generated text content.
        """

        images = kwargs.get("images")
        if images is None:
            images = []

        content: List[Dict[str, Any]] = []

        for img in images:
            image_path = Path(img)
            content.append(self._create_image_content_block(image_path))

        content.append({"type": "text", "text": prompt})

        rsp = self._run_inference(
            messages=[{"role": "user", "content": content}]
        )
        return rsp.choices[0].message.content

    def generate_propositions(
        self,
        observation: Observation,
        propose_prompt: str,
        user_name: str,
    ) -> List[PropositionItem]:
        """Generate propositions from an observation.

        Args:
            observation: The observation to generate propositions from.
            propose_prompt: The prompt template for proposing.
            user_name: Name of the user for personalization.

        Returns:
            List of generated propositions.
        """
        prompt = (
            propose_prompt
            .replace("{user_name}", user_name)
            .replace("{inputs}", observation.content)
        )

        rsp = self._run_inference(
            messages=[{"role": "user", "content": prompt}]
        )
        # schema_obj: PropositionSchema = PropositionSchema.model_validate_json(
        #     rsp.choices[0].message.content
        # )
        return parse_model_json(rsp.choices[0].message.content)