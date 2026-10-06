"""Compare a no-merge union of modality propositions against an ideal modality.

The "candidate" set is built by concatenating the propositions of the selected
modalities (``--modalities``, any subset of ax/metadata/ocr/screen) for one model
(``--candidate-model-globs``). Nothing is merged or deduplicated - this is the
no-merge baseline ("keep everything").

The candidate is then compared against an ideal/gold modality
(``--ideal-globs``, default screen_gpt-5.1) with two metric families:

* embedding-intersection metrics (reused from compare_proposition_embeddings)
* reranker classification metrics: precision / recall / f1 (reused from reranker_judge)
"""

import argparse
import csv
import sys

import numpy as np
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_proposition_embeddings import compare_proposition_sets  # noqa: E402
from scripts.proposition_io import (  # noqa: E402
    layer_flat_file,
    layer_structured_file,
    matching_folders,
    parse_user_folders,
    read_layer_flat,
    read_layer_structured,
)
from scripts.reranker_judge import evaluate_pair, write_jsonl  # noqa: E402

MODALITIES = ["ax", "metadata", "ocr", "screen"]
INPUT_LAYERS = ["propositions", "summaries", "transcriptions"]
MERGE_METHODS = [
    "none",
    "embeddings_reranker",
    "embeddings_reranker_llm",
    "sage_time_window",
    "moving_window",
]
# Arms that need the per-item time axis (read structured, with batch_id/pair_idx).
TEMPORAL_METHODS = {"sage_time_window", "moving_window"}


def method_label(merge_method: str) -> str:
    return "no_merge" if merge_method == "none" else merge_method


def uses_clustering(merge_method: str) -> bool:
    return merge_method.startswith("embeddings_reranker")


def needs_embedder(merge_method: str) -> bool:
    return uses_clustering(merge_method) or merge_method in TEMPORAL_METHODS


def build_candidate(
    user_root: Path,
    candidate_model_globs: list[str],
    modalities: list[str],
    layer: str = "propositions",
) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Concatenate (no-merge) the chosen layer's items of the selected modalities.

    Returns the combined text list, a per-modality count, and a per-modality
    record of which folder was used.
    """
    combined: list[str] = []
    per_modality_count: dict[str, int] = {}
    per_modality_folder: dict[str, str] = {}

    for modality in modalities:
        folders = matching_folders(user_root, candidate_model_globs, [modality])
        if not folders:
            per_modality_count[modality] = 0
            per_modality_folder[modality] = ""
            continue

        folder = folders[0]  # deterministic: first match by sorted name
        src_file = layer_flat_file(folder, layer)
        texts = read_layer_flat(src_file, layer) if src_file else []
        combined.extend(texts)
        per_modality_count[modality] = len(texts)
        per_modality_folder[modality] = folder.name

    return combined, per_modality_count, per_modality_folder


def build_candidate_structured(
    user_root: Path,
    candidate_model_globs: list[str],
    modalities: list[str],
    layer: str = "propositions",
) -> tuple[list[dict], dict[str, int], dict[str, str]]:
    """Like build_candidate but keeps the time axis per item (batch_id/pair_idx)."""
    items: list[dict] = []
    per_modality_count: dict[str, int] = {}
    per_modality_folder: dict[str, str] = {}

    for modality in modalities:
        folders = matching_folders(user_root, candidate_model_globs, [modality])
        if not folders:
            per_modality_count[modality] = 0
            per_modality_folder[modality] = ""
            continue

        folder = folders[0]
        src_file = layer_structured_file(folder, layer)
        props = read_layer_structured(src_file, layer) if src_file else []
        for prop in props:
            prop["modality"] = modality
        items.extend(props)
        per_modality_count[modality] = len(props)
        per_modality_folder[modality] = folder.name

    return items, per_modality_count, per_modality_folder


def evaluate_user(
    user_label: str,
    user_root: Path,
    args: argparse.Namespace,
    calculator: Any,
    reranker: Any,
    synthesizer: Any = None,
) -> dict:
    row: dict[str, Any] = {"user": user_label}

    layer = args.input_layer

    # Ideal = the union (no-merge) of all requested modalities of the ideal model.
    ideal_props, ideal_counts, ideal_folders = build_candidate(
        user_root, args.ideal_model_globs, args.ideal_modalities, layer=layer
    )

    structured_items: list[dict] = []
    if args.merge_method in TEMPORAL_METHODS:
        structured_items, per_modality_count, per_modality_folder = build_candidate_structured(
            user_root, args.candidate_model_globs, args.modalities, layer=layer
        )
        candidate_props = [item["proposition"] for item in structured_items]
    else:
        candidate_props, per_modality_count, per_modality_folder = build_candidate(
            user_root, args.candidate_model_globs, args.modalities, layer=layer
        )

    row["ideal_model"] = "+".join(f for f in ideal_folders.values() if f)
    row["ideal_modalities"] = "+".join(args.ideal_modalities)
    row["ideal_count"] = len(ideal_props)
    row["modalities"] = "+".join(args.modalities)
    for modality in args.modalities:
        row[f"count_{modality}"] = per_modality_count.get(modality, 0)
        row[f"folder_{modality}"] = per_modality_folder.get(modality, "")
    row["candidate_count"] = len(candidate_props)

    if not ideal_props:
        row["status"] = "missing_ideal"
        return row
    if not candidate_props:
        row["status"] = "empty_candidate"
        return row

    row["status"] = "ok"

    # Apply the merge arm (no-op for --merge-method none). eval_props is what
    # actually gets scored against the ideal.
    eval_props = candidate_props
    if uses_clustering(args.merge_method) and candidate_props:
        from scripts.merging import merge_embeddings_reranker

        eval_props, merge_stats = merge_embeddings_reranker(
            candidate_props,
            embedder=calculator.model,
            reranker=reranker,
            sim_threshold=args.merge_sim_threshold,
            reranker_threshold=args.merge_reranker_threshold,
            score_transform=args.score_transform,
            batch_size=args.batch_size,
            synthesize_fn=synthesizer,
        )
        row.update(merge_stats)
    elif args.merge_method == "sage_time_window" and structured_items:
        from scripts.merging import merge_sage_time_window

        eval_props, merge_stats = merge_sage_time_window(
            structured_items,
            embedder=calculator.model,
            novelty_threshold=args.novelty_threshold,
            time_window=args.time_window,
            decay_lambda=args.time_decay_lambda,
            synthesize_fn=synthesizer,
        )
        row.update(merge_stats)
    elif args.merge_method == "moving_window" and structured_items:
        from scripts.merging import merge_moving_window

        eval_props, merge_stats = merge_moving_window(
            structured_items,
            embedder=calculator.model,
            window_size=args.window_size,
            sim_threshold=args.merge_sim_threshold,
            synthesize_fn=synthesizer,
        )
        row.update(merge_stats)
    row["eval_count"] = len(eval_props)

    sims = None  # ideal x candidate cosine, reused by the judge shortlist
    if not args.skip_embeddings:
        # source = candidate (merged/union), reference = ideal (gold)
        embedding_metrics = compare_proposition_sets(
            calculator,
            eval_props,
            ideal_props,
            args.semantic_jaccard_threshold,
            return_pairwise=args.shortlist_k > 0,
            soft_low=args.soft_low,
            soft_high=args.soft_high,
        )
        pairwise = embedding_metrics.pop("_pairwise", None)
        if pairwise is not None:
            sims = np.ascontiguousarray(pairwise.T)  # -> references x predictions
        row.update(embedding_metrics)

    if not args.skip_reranker:
        # references = ideal (gold): recall = ideal covered by candidate,
        # precision = candidate propositions supported by the ideal.
        coverage_rows, precision_rows, summary = evaluate_pair(
            reranker=reranker,
            references=ideal_props,
            predictions=eval_props,
            threshold=args.score_threshold,
            top_k=args.top_k,
            batch_size=args.batch_size,
            score_transform=args.score_transform,
            progress=not args.no_progress,
            desc=f"{user_label} [{row['modalities']}] {args.merge_method} judge",
            shortlist_k=args.shortlist_k,
            sims=sims,
            embedder=calculator.model if calculator is not None else None,
        )
        row.update(
            {
                "recall_rate": summary["recall_rate"],
                "precision_rate": summary["precision_rate"],
                "f1_score": summary.get("f1_score", 0.0),
                "covered_count": summary.get("covered_count", 0),
                "supported_count": summary.get("supported_count", 0),
                "mean_best_ref_to_pred_score": summary.get("mean_best_ref_to_pred_score", 0.0),
                "mean_best_pred_to_ref_score": summary.get("mean_best_pred_to_ref_score", 0.0),
                "judge_shortlist_k": summary.get("judge_shortlist_k", 0),
                "judge_pairs_total": summary.get("judge_pairs_total", 0),
                "judge_pairs_scored": summary.get("judge_pairs_scored", 0),
                "judge_seconds": summary.get("judge_seconds", 0.0),
            }
        )

        stem = f"{method_label(args.merge_method)}_{'+'.join(args.modalities)}"
        output_dir = user_root / args.output_dir_name
        write_jsonl(output_dir / f"{stem}.coverage.jsonl", coverage_rows)
        write_jsonl(output_dir / f"{stem}.precision.jsonl", precision_rows)

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--propositions-root", default="propositions")
    parser.add_argument(
        "--user-folders",
        nargs="+",
        default=None,
        help="User folders (name, path, or label=path). Default: built-in folders.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=MODALITIES,
        help="Modalities to union (no-merge): any subset of ax/metadata/ocr/screen, or a merged-observation "
             "folder prefix such as merged+ax+metadata+ocr (see main_merged.py). Default: all 4.",
    )
    parser.add_argument(
        "--ideal-model-globs",
        nargs="+",
        default=["*gpt_5.5*", "*gpt-5.5*", "*gpt*5.5*"],
        help="Glob(s) selecting the ideal model's folders (unioned across --ideal-modalities). Default: gpt_5.5.",
    )
    parser.add_argument(
        "--ideal-modalities",
        nargs="+",
        default=MODALITIES,
        choices=MODALITIES,
        help="Modalities of the ideal model to union together. Default: all 4.",
    )
    parser.add_argument(
        "--candidate-model-globs",
        nargs="+",
        default=["*_qwen3*"],
        help="Glob(s) selecting the model whose modality propositions are unioned. Default: qwen3.",
    )
    parser.add_argument(
        "--merge-method",
        choices=MERGE_METHODS,
        default="none",
        help="How to merge the candidate union before scoring. none = no-merge baseline.",
    )
    parser.add_argument(
        "--merge-sim-threshold",
        type=float,
        default=0.75,
        help="Embedding cosine threshold for recalling merge candidates. Default: 0.75",
    )
    parser.add_argument(
        "--merge-reranker-threshold",
        type=float,
        default=0.5,
        help="Reranker score threshold to confirm a merge. Default: 0.5",
    )
    parser.add_argument(
        "--llm-merge-model",
        default="gpt-4o-mini",
        help="OpenAI model used to synthesize merged propositions (embeddings_reranker_llm). Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--llm-merge-temperature",
        type=float,
        default=0.0,
        help="Temperature for the LLM merge model. Default: 0.0",
    )
    parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.8,
        help="SAGE novelty gate: absorb if in-window cosine >= this. Default: 0.8",
    )
    parser.add_argument(
        "--time-window",
        type=lambda v: None if v.lower() == "none" else int(v),
        default=2,
        help="SAGE: only merge memories within this many batch_ids; 'none' = no hard window. Default: 2",
    )
    parser.add_argument(
        "--time-decay-lambda",
        type=float,
        default=None,
        help="SAGE: optional exp(-Δbatch/lambda) similarity decay. Default: off (hard window).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="moving_window: tumbling window size in time units (batch_id/pair_idx). Default: 5",
    )
    parser.add_argument(
        "--moving-window-llm",
        action="store_true",
        help="moving_window: synthesize merged text with the LLM instead of keeping the medoid.",
    )
    parser.add_argument(
        "--input-layer",
        choices=INPUT_LAYERS,
        default="propositions",
        help="Which layer to merge/score: propositions, summaries, or transcriptions. Default: propositions",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="auto", help="Torch device: auto/cuda/mps/cpu. Default: auto")
    parser.add_argument("--semantic-jaccard-threshold", type=float, default=0.75)
    parser.add_argument("--soft-low", type=float, default=0.60,
                        help="Soft-coverage ramp start (cosine); calibrate with scripts/calibrate_soft_metric.py. Default: 0.60")
    parser.add_argument("--soft-high", type=float, default=0.80,
                        help="Soft-coverage ramp end (cosine). Default: 0.80")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--score-transform", choices=["none", "sigmoid"], default="sigmoid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--shortlist-k",
        type=int,
        default=0,
        help="Rerank only each ideal proposition's top-k candidates by embedding cosine (and vice "
             "versa) instead of the full product. 0 = off, full matrix. Reuses the cosine matrix "
             "the embedding metrics already build, so the shortlist itself costs nothing.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm reranker-judge progress bar.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding-intersection metrics.")
    parser.add_argument("--skip-reranker", action="store_true", help="Skip reranker classification metrics.")
    parser.add_argument(
        "--output-dir-name",
        default=None,
        help="Output dir inside each user folder for reranker JSONL. Default: <method>_vs_ideal_results.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Summary CSV path. Default: propositions/<method>_modalities_vs_ideal.csv.",
    )
    args = parser.parse_args()

    label = method_label(args.merge_method)
    if args.output_dir_name is None:
        args.output_dir_name = f"{label}_vs_ideal_results"
    if args.output is None:
        args.output = f"propositions/{label}_modalities_vs_ideal.csv"

    # Clustering merges need both models; SAGE needs only the embedder.
    root = Path(args.propositions_root)
    users = parse_user_folders(args.user_folders, root)

    calculator = None
    if not args.skip_embeddings or needs_embedder(args.merge_method):
        from src.user_profile_similarity import UserProfileSimilarity

        print(f"Loading embedding model: {args.embedding_model}...")
        calculator = UserProfileSimilarity(model_name=args.embedding_model, device=args.device)

    reranker = None
    if not args.skip_reranker or uses_clustering(args.merge_method):
        from sentence_transformers import CrossEncoder

        from src.models.utils.device import resolve_torch_device

        device = resolve_torch_device(args.device)
        print(f"Loading reranker model: {args.reranker_model} on {device}...")
        reranker = CrossEncoder(args.reranker_model, device=device)

    synthesizer = None
    use_llm_synthesis = args.merge_method == "embeddings_reranker_llm" or (
        args.merge_method == "moving_window" and args.moving_window_llm
    )
    if use_llm_synthesis:
        from scripts.llm_merge import build_client, make_synthesizer

        print(f"Using OpenAI model for LLM merge: {args.llm_merge_model}")
        synthesizer = make_synthesizer(
            build_client(), model=args.llm_merge_model, temperature=args.llm_merge_temperature
        )

    rows = []
    for user_label, user_root in users.items():
        row = evaluate_user(user_label, user_root, args, calculator, reranker, synthesizer)
        rows.append(row)

        if row["status"] != "ok":
            print(f"{user_label} [{row['modalities']}]: {row['status']}")
            continue
        msg = (
            f"{user_label} [{row['modalities']}]: "
            f"candidate={row['candidate_count']}"
        )
        if args.merge_method != "none":
            msg += f" -> merged={row.get('eval_count', row['candidate_count'])}"
        msg += f" vs ideal={row['ideal_count']}"
        if not args.skip_embeddings:
            msg += f" | symmetric_best_mean={row['symmetric_best_mean']:.4f}, semantic_jaccard={row['semantic_jaccard']:.4f}"
        if not args.skip_reranker:
            msg += f" | precision={row['precision_rate']:.4f}, recall={row['recall_rate']:.4f}, f1={row['f1_score']:.4f}"
        print(msg)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
