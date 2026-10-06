import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import platform

IS_MACOS = platform.system() == "Darwin"
if IS_MACOS:
    from src.models.mlx_model import MLXVisionModel

from src.providers.screen_provider import ScreenProvider
from src.proposition_generator import PropositionGenerator
from src.data_preprocessors.frames_preprocessor import save_stable_frames
from src.helpers.convert_propositions import convert_propositions_to_csv
from src.user_profile_generator import generate_user_profile
from src.cli_utils import default_experiment_id, resolve_provider
from src.timestamp_enrichment.context import TimestampContext, modality_for_pipeline

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the screen-recording user-understanding pipeline.")

    # Experiment identity (required).
    parser.add_argument("--user-name", required=True, help="User/experiment name.")
    parser.add_argument("--model-name", required=True, help="Model identifier (must be listed in SUPPORTED_MODELS).")
    parser.add_argument("--paths-to-videos", nargs="+", required=True, help="One or more video paths.")

    # Output layout.
    parser.add_argument("--experiment-prefix", default="screen", help="Default: screen")
    parser.add_argument("--output-dir", default="output_data", help="Default: output_data")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Full experiment output path. If omitted, derived from output dir, user name, prefix, and model name.",
    )
    parser.add_argument(
        "--path-to-frames",
        default=None,
        help="Default: {output_dir}/{user_name}_input_frames",
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
    parser.add_argument("--diff-threshold", type=float, default=0.5, help="Stable-frame diff threshold. Default: 0.5")
    parser.add_argument("--stability-duration", type=float, default=1.0, help="Stable-frame duration in seconds. Default: 1.0")
    parser.add_argument(
        "--allowed-local-media-path",
        default=None,
        help="Directory vLLM may read local media from. Default: resolved frame directory. Only used with vLLM models.",
    )
    parser.add_argument(
        "--apply-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply OCR before screen transcription and summarization.",
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
    paths_to_videos = args.paths_to_videos
    path_to_frames = args.path_to_frames or os.path.join(args.output_dir, f"{args.user_name}_input_frames")

    params = {
        "user_name": args.user_name,
        "model_name": args.model_name,
        "experiment_id": experiment_id,
        "paths_to_videos": paths_to_videos,
        "path_to_frames": path_to_frames,
        "history_size": args.history_size,
        "k_items_per_batch": args.k_items_per_batch,
        "chunk_size": args.chunk_size,
        "diff_threshold": args.diff_threshold,
        "stability_duration": args.stability_duration,
        "apply_ocr": args.apply_ocr,
        "enrich_timestamps": args.enrich_timestamps,
    }
    print("=== Run parameters ===")
    for key, value in params.items():
        print(f"  {key:<18}: {value}")
    print("======================")

    save_stable_frames(
        video_paths=paths_to_videos,
        output_dir=path_to_frames,
        diff_threshold=args.diff_threshold,
        stability_duration=args.stability_duration,
    )

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
            allowed_local_media_path=args.allowed_local_media_path or path_to_frames,
        )
    else:
        raise ValueError(f"Model {args.model_name} is not supported.")

    modality = modality_for_pipeline(args.experiment_prefix, apply_ocr=args.apply_ocr)
    timestamp_ctx = None
    if args.enrich_timestamps:
        timestamp_ctx = TimestampContext.build(
            modality,
            events_dir=Path(paths_to_videos[0]).parent,
            frames_dir=path_to_frames,
        )

    screen_provider = ScreenProvider(
        model=model,
        path_to_store_experiment=experiment_id,
        history_size=args.history_size,
        apply_ocr=args.apply_ocr,
        timestamp_ctx=timestamp_ctx,
        chunk_size=args.chunk_size,
    )

    screen_provider.process_frames(
        frames_dir=path_to_frames
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
