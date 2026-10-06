import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import platform

IS_MACOS = platform.system() == "Darwin"
if IS_MACOS:
    from src.models.mlx_model import MLXVisionModel

from src.providers.ax_provider import AXProvider
from src.proposition_generator import PropositionGenerator
from src.data_preprocessors.ax_preprocessor import AXProcessor
from src.helpers.convert_propositions import convert_propositions_to_csv
from src.user_profile_generator import generate_user_profile
from src.cli_utils import default_experiment_id, resolve_provider
from src.timestamp_enrichment.context import TimestampContext, modality_for_pipeline

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the accessibility-tree user-understanding pipeline.")

    # Experiment identity (required).
    parser.add_argument("--user-name", required=True, help="User/experiment name.")
    parser.add_argument("--model-name", required=True, help="Model identifier (must be listed in SUPPORTED_MODELS).")
    parser.add_argument(
        "--path-to-accessibility",
        required=True,
        help="Path to the raw accessibility.json file (app_events.json must sit beside it for timestamps).",
    )

    # Output layout.
    parser.add_argument("--experiment-prefix", default="ax", help="Default: ax")
    parser.add_argument("--output-dir", default="output_data", help="Default: output_data")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Full experiment output path. If omitted, derived from output dir, user name, prefix, and model name.",
    )
    parser.add_argument(
        "--path-to-clean-accessibility",
        default=None,
        help="Default: {output_dir}/{user_name}_accessibility_clean.json",
    )

    # Processing parameters.
    parser.add_argument(
        "--k-items-per-batch",
        type=int,
        default=5,
        help="Number of transcription/summary pairs per proposition-generation batch. Default: 5",
    )
    parser.add_argument("--history-size", type=int, default=2, help="Default: 2")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Requests submitted per batched model call (SGLang only). "
             "Keep >= data_parallel_size so every GPU stays busy. Default: 3",
    )
    parser.add_argument(
        "--max-model-input-context-size",
        type=int,
        default=12_000,
        help="Default: 12000",
    )
    parser.add_argument(
        "--allowed-local-media-path",
        default=None,
        help="Directory vLLM may read local media from. Only used with vLLM models.",
    )
    parser.add_argument(
        "--enrich-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach timestamp_range to pipeline outputs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    experiment_id = args.experiment_id or default_experiment_id(
        args.output_dir, args.user_name, args.experiment_prefix, args.model_name
    )
    path_to_accessibility = args.path_to_accessibility
    path_to_clean_accessibility = (
        args.path_to_clean_accessibility
        or os.path.join(args.output_dir, f"{args.user_name}_accessibility_clean.json")
    )

    params = {
        "user_name": args.user_name,
        "model_name": args.model_name,
        "experiment_id": experiment_id,
        "path_to_accessibility": path_to_accessibility,
        "path_to_clean_accessibility": path_to_clean_accessibility,
        "history_size": args.history_size,
        "max_model_input_context_size": args.max_model_input_context_size,
        "k_items_per_batch": args.k_items_per_batch,
        "chunk_size": args.chunk_size,
        "enrich_timestamps": args.enrich_timestamps,
    }
    print("=== Run parameters ===")
    for key, value in params.items():
        print(f"  {key:<28}: {value}")
    print("======================")

    AXProcessor.process_and_save(path_to_accessibility, path_to_clean_accessibility)

    if os.path.exists(experiment_id):
        print(f"[ERROR] Experiment '{experiment_id}' already exists.")
        return

    os.makedirs(experiment_id, exist_ok=True)
    with open(os.path.join(experiment_id, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    provider = resolve_provider(args.model_name)

    if provider == "openai":
        from src.models.openai_model import OpenAIVisionModel

        model = OpenAIVisionModel(
            model_identifier=args.model_name,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "mlx":
        model = MLXVisionModel(
            model_identifier=args.model_name
        )
    elif provider == "mlx_lm":
        from src.models.mlx_lm_model import MLXModel

        model = MLXModel(
            model_identifier=args.model_name
        )
    elif provider == "sglang":
        from src.models.sglang_model import SGLangVisionModel

        model = SGLangVisionModel(
            model_identifier=args.model_name,
            allowed_local_media_path=args.allowed_local_media_path,
        )
    else:
        raise ValueError(f"Model {args.model_name} is not supported.")

    modality = modality_for_pipeline(args.experiment_prefix)
    timestamp_ctx = None
    if args.enrich_timestamps:
        timestamp_ctx = TimestampContext.build(
            modality,
            events_dir=Path(path_to_accessibility).parent,
        )

    ax_provider = AXProvider(
        model=model,
        path_to_store_experiment=experiment_id,
        model_context_size=args.max_model_input_context_size,
        history_size=args.history_size,
        timestamp_ctx=timestamp_ctx,
        chunk_size=args.chunk_size,
    )
    ax_provider.process_ax(
        ax_path=path_to_clean_accessibility,
    )

    generator = PropositionGenerator(
        model=model,
        user_name=args.user_name,
        k_items_per_batch=args.k_items_per_batch,
        chunk_size=args.chunk_size,
        log_filename=f"{experiment_id}/logs.log"
    )

    generator.process(
        summaries_path=f"{experiment_id}/summaries.jsonl",
        transcriptions_path=f"{experiment_id}/transcriptions.jsonl",
        output_path=f"{experiment_id}/propositions.jsonl",
        enrich_timestamps=args.enrich_timestamps,
        modality=modality if args.enrich_timestamps else None,
    )

    convert_propositions_to_csv(f"{experiment_id}/propositions.jsonl", f"{experiment_id}/propositions.csv")
    generate_user_profile(f"{experiment_id}/propositions.csv", f"{experiment_id}/user_profile.json")


if __name__ == "__main__":
    main()
