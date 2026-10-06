"""LLM-based proposition cluster merging via the OpenAI API.

Given a cluster of near-duplicate propositions (all describing the same user,
possibly extracted from different modalities), ask an LLM to synthesize ONE
proposition that captures them and label the merge_type
(generalize / refine / supersede / contradict).

Used as the cluster reducer for the ``embeddings_reranker_llm`` merge arm, and
runnable standalone for quick checks:

    python scripts/llm_merge.py --propositions "A ..." "B ..." "C ..."
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MERGE_TYPES = ["generalize", "refine", "supersede", "contradict"]

# Models that reject an explicit temperature (mirrors src.configs.general_config).
FIXED_TEMPERATURE_MODELS = {"gpt-5.1", "gpt-5.5", "o1", "o1-mini", "o3", "o3-mini"}

SYSTEM_MSG = (
    "You merge several near-duplicate user propositions into a SINGLE proposition. "
    "The propositions all describe the same user and were extracted from different "
    "modalities (screen, OCR, accessibility, metadata).\n\n"
    "Write one clear, self-contained proposition that captures what the cluster asserts. "
    "Do NOT invent facts that are not supported by the inputs. Prefer the most specific "
    "wording that still covers every input.\n\n"
    "Also label the merge_type:\n"
    "- generalize: inputs are variants of the same idea -> one broader claim\n"
    "- refine: one input is a more precise version -> keep the precise wording\n"
    "- supersede: a stronger/later input overrides a weaker/older one\n"
    "- contradict: inputs conflict -> state the most supported claim and note the conflict"
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "merged_proposition",
        "schema": {
            "type": "object",
            "properties": {
                "merged_proposition": {"type": "string"},
                "merge_type": {"type": "string", "enum": MERGE_TYPES},
            },
            "required": ["merged_proposition", "merge_type"],
            "additionalProperties": False,
        },
    },
}


def build_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


def merge_propositions(
    cluster: list[str],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> dict:
    """Merge a cluster of propositions into one. Returns {merged_proposition, merge_type}."""
    if len(cluster) <= 1:
        return {"merged_proposition": cluster[0] if cluster else "", "merge_type": "refine"}

    user_msg = "Merge these propositions into one:\n" + json.dumps(cluster, ensure_ascii=False, indent=2)
    kwargs: dict = {
        "model": model,
        "response_format": RESPONSE_FORMAT,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
    }
    if model not in FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)
    return json.loads(resp.choices[0].message.content)


def make_synthesizer(client: OpenAI, model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Return a callable ``cluster_texts -> merged_text`` for use as a cluster reducer."""

    def _synthesize(cluster_texts: list[str]) -> str:
        return merge_propositions(cluster_texts, client, model=model, temperature=temperature)[
            "merged_proposition"
        ]

    return _synthesize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--propositions", nargs="+", help="A single cluster of propositions to merge.")
    parser.add_argument("--input-json", help="Path to a JSON file with a list of clusters (list[list[str]]).")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    client = build_client()

    if args.input_json:
        clusters = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        for cluster in clusters:
            print(json.dumps(merge_propositions(cluster, client, args.model, args.temperature), ensure_ascii=False))
    elif args.propositions:
        result = merge_propositions(args.propositions, client, args.model, args.temperature)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.error("provide --propositions or --input-json")


if __name__ == "__main__":
    main()
