from typing import Optional, List
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field, ConfigDict

class PropositionItem(BaseModel):
    reasoning: str = Field(..., description="The reasoning for the proposition")
    proposition: str = Field(..., description="The proposition string")
    confidence: int = Field(
        ...,
        ge=1,
        le=10,
        description="Confidence score from 1 (low) to 10 (high)"
    )
    decay: int = Field(
        ...,
        ge=1,
        le=10,
        description="Decay score from 1 (low) to 10 (high)"
    )
    model_config = ConfigDict(extra="forbid")

class PropositionSchema(BaseModel):
    propositions: List[PropositionItem] = Field(
        ...,
        description="Up to K propositions"
    )
    model_config = ConfigDict(extra="forbid")

class Observation(BaseModel):
    content: str
    pair_indices: List[int]

# TODO: move to separate file
@dataclass
class Session:
    id: int
    start: str
    end: str
    app_event: dict
    kb_events: list[dict]
    mouse_events: list[dict]

@dataclass
class MetadataEvent:
    id: int
    duration: float
    trigger_action: str
    app_name: str
    windows_names: str
    mouse_metrics: Optional[dict] = None
    typing_metrics: Optional[dict] = None
    keyboard_transcription: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class GenerationStats:
    prompt_tokens: int
    generation_tokens: int
    total_tokens: int
    latency_sec: float
    prompt_tps: float
    generation_tps: float
    peak_memory: float

    def to_log_row(self) -> str:
        return (
            f"prompt_tokens={self.prompt_tokens}, "
            f"generation_tokens={self.generation_tokens}, "
            f"total_tokens={self.total_tokens}, "
            f"latency_sec={self.latency_sec:.4f}, "
            f"prompt_tps={self.prompt_tps:.2f}, "
            f"generation_tps={self.generation_tps:.2f}, "
            f"peak_memory={self.peak_memory:.2f}"
        )
