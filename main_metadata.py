import os
import json
import argparse
from dotenv import load_dotenv
import platform

IS_MACOS = platform.system() == "Darwin"
if IS_MACOS:
    from src.models.mlx_model import MLXVisionModel

from src.providers.metadata_provider import MetadataProvider
from src.proposition_generator import PropositionGenerator
from src.data_preprocessors.metadata_preprocessor.metadata_preprocessor import MetadataPreprocessor
from src.helpers.convert_propositions import convert_propositions_to_csv
from src.user_profile_generator import generate_user_profile
from src.cli_utils import default_experiment_id, resolve_provider
from src.timestamp_enrichment.context import TimestampContext, modality_for_pipeline

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the metadata user-understanding pipeline.")

    # Experiment identity (required).
    parser.add_argument("--user-name", required=True, help="User/experiment name.")
    parser.add_argument("--model-name", required=True, help="Model identifier (must be listed in SUPPORTED_MODELS).")
    parser.add_argument(
        "--path-to-metadata",
        required=True,
        help="Directory holding app_events.json / mouse_events.json / keyboard_events.json.",
    )

    # Output layout.
    parser.add_argument("--experiment-prefix", default="metadata", help="Default: metadata")
    parser.add_argument("--output-dir", default="output_data", help="Default: output_data")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Full experiment output path. If omitted, derived from output dir, user name, prefix, and model name.",
    )
    parser.add_argument(
        "--path-to-clean-metadata",
        default=None,
        help="Default: {output_dir}/{user_name}_metadata_clean.json",
    )

    # Processing parameters.
    parser.add_argument(
        "--k-items-per-batch",
        type=int,
        default=5,
        help="Number of transcription/summary pairs per proposition-generation batch. Default: 5",
    )
    parser.add_argument("--history-size", type=int, default=10, help="Default: 10")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Requests submitted per batched model call (SGLang only). "
             "Keep >= data_parallel_size so every GPU stays busy. Default: 3",
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
    path_to_metadata = args.path_to_metadata
    path_to_clean_metadata = (
        args.path_to_clean_metadata
        or os.path.join(args.output_dir, f"{args.user_name}_metadata_clean.json")
    )

    params = {
        "user_name": args.user_name,
        "model_name": args.model_name,
        "experiment_id": experiment_id,
        "path_to_metadata": path_to_metadata,
        "path_to_clean_metadata": path_to_clean_metadata,
        "history_size": args.history_size,
        "k_items_per_batch": args.k_items_per_batch,
        "chunk_size": args.chunk_size,
        "enrich_timestamps": args.enrich_timestamps,
    }
    print("=== Run parameters ===")
    for key, value in params.items():
        print(f"  {key:<22}: {value}")
    print("======================")

    MetadataPreprocessor().process(path_to_metadata, path_to_clean_metadata)

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
            events_dir=path_to_metadata,
        )

    metadata_provider = MetadataProvider(
        model=model,
        path_to_store_experiment=experiment_id,
        history_size=args.history_size,
        timestamp_ctx=timestamp_ctx,
        chunk_size=args.chunk_size,
    )
    metadata_provider.process_metadata(
        metadata_path=path_to_clean_metadata
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
