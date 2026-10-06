import os
import json
import base64
import requests
from typing import List
from src.data_models import Observation, PropositionItem, PropositionSchema
from .base_model import BaseVisionModel
from src.helpers.json_parsing import parse_model_json

class LMStudioVisionModel(BaseVisionModel):
    """
    Simple LM Studio Vision-Language model client.
    Works with a local LM Studio API server
    """

    def __init__(self, model_identifier: str = "qwen/qwen3-vl-4b", base_url: str = "http://127.0.0.1:1234"):
        self.base_url = base_url
        self.client = None
        super().__init__(model_identifier)
        
    def _load_model(self):
        """Just confirm connection to LM Studio server."""
        print(f"Initializing LM Studio client for model: {self.model_identifier}...")
        self.client = self.base_url.rstrip("/")
        print(f"LM Studio client initialized at {self.client}")

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Convert an image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_response(self, images: List[str], prompt: str, **kwargs) -> str:
        """
        Send a text + images request to LM Studio and get the model response.
        """
        # print("Preparing request for LM Studio...")

        # Build message content
        content = []
        for img_path in images:
            ext = os.path.splitext(img_path)[1].lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg"
            }.get(ext, "image/jpeg")
            img_b64 = self.encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"}
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model_identifier,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4096
        }

        url = f"{self.client}/v1/chat/completions"
        # print(f"Sending request to {url} ...")

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            # print("Response received successfully.")
            return text
        except Exception as e:
            print(f"Error while generating response: {e}")
            return f"[Error] {e}"

    def generate_propositions(self, observation: Observation, propose_prompt: str, user_name: str) -> list[PropositionItem]:
        """
        Generate propositions using LM Studio.
        """
        print("Generating propositions...")

        prompt = (
            propose_prompt
            .replace("{user_name}", user_name)
            .replace("{inputs}", observation.content)
        )

        schema = PropositionSchema.model_json_schema()

        payload = {
            "model": self.model_identifier,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "proposition_schema",
                    "strict": True,
                    "schema": schema
                }
            }
        }

        url = f"{self.client}/v1/chat/completions"
        # print(f"Sending JSON request to {url} ...")

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            result = parse_model_json(data["choices"][0]["message"]["content"])
            # print("Propositions generated successfully.")
            return result
        except Exception as e:
            print(f"Error while generating propositions: {e}")
            return []
