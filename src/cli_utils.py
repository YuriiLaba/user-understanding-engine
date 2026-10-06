import argparse
import os
from typing import Any

from src.configs.general_config import SUPPORTED_MODELS


def model_slug(model_name: str) -> str:
    """Return a filesystem-friendly model name."""
    name = model_name.split("/")[-1]
    return name.lower().replace("-", "_")


def default_experiment_id(output_dir: str, user_name: str, experiment_prefix: str, model_name: str) -> str:
    return os.path.join(output_dir, user_name, f"{experiment_prefix}_{model_slug(model_name)}")


def add_common_experiment_args(
    parser: argparse.ArgumentParser,
    config: Any,
    model_default: str | None = None,
) -> None:
    model_help_default = model_default or config.MODEL_NAME
    parser.add_argument("--user-name", default=None, help=f"Default: {config.USER_NAME}")
    parser.add_argument("--model-name", default=None, help=f"Default: {model_help_default}")
    parser.add_argument("--experiment-prefix", default=None, help=f"Default: {config.EXPERIMENT_PREFIX}")
    parser.add_argument("--output-dir", default=None, help=f"Default: {config.OUTPUT_DIR}")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Full experiment output path. If omitted, derived from output dir, user name, prefix, and model name.",
    )
    parser.add_argument(
        "--k-items-per-batch",
        type=int,
        default=5,
        help="Number of transcription/summary pairs per proposition-generation batch. Default: 5",
    )


def resolve_common_args(args: argparse.Namespace, config: Any) -> dict:
    user_name = args.user_name or config.USER_NAME
    model_name = args.model_name or config.MODEL_NAME
    experiment_prefix = args.experiment_prefix or config.EXPERIMENT_PREFIX
    output_dir = args.output_dir or config.OUTPUT_DIR
    experiment_id = (
        args.experiment_id
        or default_experiment_id(output_dir, user_name, experiment_prefix, model_name)
    )
    return {
        "user_name": user_name,
        "model_name": model_name,
        "experiment_prefix": experiment_prefix,
        "output_dir": output_dir,
        "experiment_id": experiment_id,
        "k_items_per_batch": args.k_items_per_batch,
    }


def resolve_provider(model_name: str) -> str | None:
    return next((provider for provider, models in SUPPORTED_MODELS.items() if model_name in models), None)
