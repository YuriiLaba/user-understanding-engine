import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (and surrounding whitespace)."""
    return _THINK_RE.sub("", text).strip()


def parse_model_json(text: str) -> Any:
    """Parse a model's text output into JSON, tolerating <think> blocks and ```json fences.

    Raises ValueError (including the raw output) if no valid JSON can be extracted, so
    callers get a legible message instead of a bare JSONDecodeError.
    """
    cleaned = strip_think(text)

    # Drop a surrounding ```json ... ``` markdown fence if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        cleaned = re.sub(r"^json\b", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Fallback: extract the outermost {...} object and retry.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"Model did not return valid JSON ({e}). Raw output:\n{text}"
        ) from e
