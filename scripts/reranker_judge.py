import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is optional
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proposition_io import (
    first_existing_file,
    first_matching_folder,
    matching_folders,
    modality_from_folder,
    parse_user_folders,
    read_propositions,
)  # noqa: E402


def batched(items: list[tuple[str, str]], batch_size: int) -> Iterable[list[tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def transform_scores(scores: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return scores.astype(float)
    if transform == "sigmoid":
        return 1.0 / (1.0 + np.exp(-scores.astype(float)))
    raise ValueError(f"Unsupported score transform: {transform}")


def shortlist_mask(sims: np.ndarray, k: int) -> np.ndarray:
    """Pairs worth reranking: each reference's top-k predictions by cosine, and vice versa.

    ``top_matches`` only ever keeps a handful of matches per row and per column, so a pair
    outside both top-k lists cannot change coverage or precision unless the reranker and the
    embedder disagree wildly about it. Union of both directions, so recall and precision each
    keep their own candidates.
    """
    references, predictions = sims.shape
    mask = np.zeros(sims.shape, dtype=bool)
    kr, kc = min(k, predictions), min(k, references)
    row_top = np.argpartition(-sims, kr - 1, axis=1)[:, :kr]
    np.put_along_axis(mask, row_top, True, axis=1)
    col_top = np.argpartition(-sims, kc - 1, axis=0)[:kc, :]
    np.put_along_axis(mask, col_top, True, axis=0)
    return mask


def cosine_matrix(embedder: Any, references: list[str], predictions: list[str]) -> np.ndarray:
    """references x predictions cosine, for the shortlist."""
    ref = np.asarray(embedder.encode(references, show_progress_bar=False), dtype=np.float32)
    pred = np.asarray(embedder.encode(predictions, show_progress_bar=False), dtype=np.float32)
    ref /= np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)
    pred /= np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12, None)
    return np.nan_to_num(ref @ pred.T, nan=0.0, posinf=0.0, neginf=0.0)


def score_matrix(
    reranker: Any,
    references: list[str],
    predictions: list[str],
    batch_size: int,
    score_transform: str,
    progress: bool = False,
    desc: str = "",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if mask is not None:
        # Only the shortlisted pairs reach the cross-encoder; the rest stay at -inf, which is
        # below any threshold, so they can never be counted as a match.
        rows, cols = np.nonzero(mask)
        pairs = [(references[i], predictions[j]) for i, j in zip(rows, cols)]
        flat = _predict(reranker, pairs, batch_size, progress, desc)
        matrix = np.full((len(references), len(predictions)), -np.inf, dtype=float)
        matrix[rows, cols] = transform_scores(flat, score_transform)
        return matrix
    pairs = [(ref, pred) for ref in references for pred in predictions]
    # Pass batch_size through to the CrossEncoder so it actually reaches the GPU
    # (without it, predict() falls back to its internal default of 32).
    if progress and tqdm is not None:
        scores: list[float] = []
        bar = tqdm(total=len(pairs), desc=desc or "reranker judge", unit="pair", leave=False)
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            chunk_scores = reranker.predict(chunk, batch_size=batch_size, show_progress_bar=False)
            scores.extend(np.asarray(chunk_scores).reshape(-1).tolist())
            bar.update(len(chunk))
        bar.close()
        flat = np.asarray(scores, dtype=float)
    else:
        flat = np.asarray(
            reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=float
        )
    matrix = flat.reshape(len(references), len(predictions))
    return transform_scores(matrix, score_transform)


def _predict(reranker: Any, pairs: list[tuple[str, str]], batch_size: int, progress: bool, desc: str) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=float)
    if progress and tqdm is not None:
        # Feed predict() a slab of many batches at a time rather than one batch per call:
        # it sorts its input by length internally, so bigger slabs mean less padding, and it
        # keeps the progress bar roughly as responsive.
        slab = max(batch_size * 16, batch_size)
        out = np.empty(len(pairs), dtype=float)
        bar = tqdm(total=len(pairs), desc=desc or "reranker judge", unit="pair", leave=False)
        for start in range(0, len(pairs), slab):
            chunk = pairs[start : start + slab]
            out[start : start + len(chunk)] = np.asarray(
                reranker.predict(chunk, batch_size=batch_size, show_progress_bar=False)
            ).reshape(-1)
            bar.update(len(chunk))
        bar.close()
        return out
    return np.asarray(reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=float).reshape(-1)


def top_matches(scores: np.ndarray, threshold: float, top_k: int) -> list[tuple[int, float]]:
    if scores.size == 0:
        return []
    indices = np.argsort(scores)[::-1]
    matches = []
    for idx in indices[:top_k]:
        score = float(scores[idx])
        if score < threshold:
            continue
        matches.append((int(idx), score))
    return matches


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_pair(
    reranker: Any,
    references: list[str],
    predictions: list[str],
    threshold: float,
    top_k: int,
    batch_size: int,
    score_transform: str,
    progress: bool = False,
    desc: str = "",
    shortlist_k: int = 0,
    sims: np.ndarray | None = None,
    embedder: Any = None,
) -> tuple[list[dict], list[dict], dict]:
    if not references or not predictions:
        return [], [], {
            "recall_rate": 0.0,
            "precision_rate": 0.0,
            "reference_count": len(references),
            "prediction_count": len(predictions),
        }

    mask = None
    if shortlist_k > 0:
        if sims is None:
            if embedder is None:
                raise ValueError("shortlist_k needs either sims or an embedder")
            sims = cosine_matrix(embedder, references, predictions)
        mask = shortlist_mask(sims, shortlist_k)

    started = time.perf_counter()
    matrix = score_matrix(reranker, references, predictions, batch_size, score_transform,
                          progress, desc, mask=mask)
    judge_seconds = time.perf_counter() - started

    coverage_rows = []
    for ref_id, ref_scores in enumerate(matrix):
        matches = top_matches(ref_scores, threshold, top_k)
        coverage_rows.append(
            {
                "ref_id": ref_id,
                "coverage": "COVERED" if matches else "NOT_COVERED",
                "matched_pred_ids": [idx for idx, _ in matches],
                "matched_scores": [round(score, 6) for _, score in matches],
                "evidence_snippets": [predictions[idx] for idx, _ in matches],
                "notes": f"reranker_score_threshold={threshold}",
            }
        )

    precision_rows = []
    for pred_id, pred_scores in enumerate(matrix.T):
        matches = top_matches(pred_scores, threshold, top_k)
        precision_rows.append(
            {
                "pred_id": pred_id,
                "label": "SUPPORTED" if matches else "UNSUPPORTED",
                "support_ref_ids": [idx for idx, _ in matches],
                "matched_scores": [round(score, 6) for _, score in matches],
                "evidence_snippets": [references[idx] for idx, _ in matches],
                "notes": f"reranker_score_threshold={threshold}",
            }
        )

    covered = sum(row["coverage"] == "COVERED" for row in coverage_rows)
    supported = sum(row["label"] == "SUPPORTED" for row in precision_rows)
    recall_rate = covered / len(coverage_rows) if coverage_rows else 0.0
    precision_rate = supported / len(precision_rows) if precision_rows else 0.0
    f1_score = (
        2 * precision_rate * recall_rate / (precision_rate + recall_rate)
        if precision_rate + recall_rate > 0
        else 0.0
    )
    summary = {
        "recall_rate": recall_rate,
        "precision_rate": precision_rate,
        "f1_score": f1_score,
        "reference_count": len(references),
        "prediction_count": len(predictions),
        "covered_count": covered,
        "supported_count": supported,
        "mean_best_ref_to_pred_score": float(np.max(matrix, axis=1).mean()),
        "mean_best_pred_to_ref_score": float(np.max(matrix, axis=0).mean()),
        "judge_shortlist_k": shortlist_k,
        "judge_pairs_total": len(references) * len(predictions),
        "judge_pairs_scored": int(mask.sum()) if mask is not None else len(references) * len(predictions),
        "judge_seconds": round(judge_seconds, 3),
    }
    return coverage_rows, precision_rows, summary


def main():
    parser = argparse.ArgumentParser(
        description="Reranker-based replacement for LLM-as-judge proposition coverage/precision classification."
    )
    parser.add_argument("--propositions-root", default="propositions")
    parser.add_argument(
        "--user-folders",
        nargs="+",
        default=None,
        help=(
            "User proposition folders. Values can be names relative to --propositions-root, "
            "paths, or label=path. Default: built-in folders."
        ),
    )
    parser.add_argument(
        "--reference-model-globs",
        nargs="+",
        default=["screen_gpt*5.1", "screen_gpt_5.1", "screen_gpt-5.1"],
        help="Glob(s) for the reference/baseline folder inside each user folder.",
    )
    parser.add_argument(
        "--prediction-model-globs",
        nargs="+",
        default=["*_qwen3*"],
        help="Glob(s) for prediction folders inside each user folder.",
    )
    parser.add_argument("--modes", nargs="+", default=None, help="Optional modality filter, e.g. screen metadata.")
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="SentenceTransformers CrossEncoder reranker model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the CrossEncoder reranker. Use auto to prefer CUDA, then MPS, then CPU. Default: auto",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--score-transform", choices=["none", "sigmoid"], default="sigmoid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir-name",
        default="reranker_as_a_judge_results",
        help="Output directory name created inside each user folder.",
    )
    args = parser.parse_args()

    root = Path(args.propositions_root)
    users = parse_user_folders(args.user_folders, root)
    from src.models.utils.device import resolve_torch_device

    device = resolve_torch_device(args.device)
    from sentence_transformers import CrossEncoder

    print(f"Loading reranker model: {args.reranker_model} on {device}...")
    reranker = CrossEncoder(args.reranker_model, device=device)
    summary_rows = []

    for user_label, user_root in users.items():
        reference_folder = first_matching_folder(user_root, args.reference_model_globs)
        reference_file = first_existing_file(reference_folder) if reference_folder else None
        prediction_folders = matching_folders(user_root, args.prediction_model_globs, args.modes)

        if not reference_file:
            print(f"{user_label}: missing reference folder/file")
            continue

        references = read_propositions(reference_file)
        output_dir = user_root / args.output_dir_name

        for prediction_folder in prediction_folders:
            prediction_file = first_existing_file(prediction_folder)
            if not prediction_file:
                print(f"{user_label} {prediction_folder.name}: missing propositions file")
                continue

            predictions = read_propositions(prediction_file)
            coverage_rows, precision_rows, summary = evaluate_pair(
                reranker=reranker,
                references=references,
                predictions=predictions,
                threshold=args.score_threshold,
                top_k=args.top_k,
                batch_size=args.batch_size,
                score_transform=args.score_transform,
            )

            stem = prediction_folder.name
            write_jsonl(output_dir / f"{stem}.coverage.jsonl", coverage_rows)
            write_jsonl(output_dir / f"{stem}.precision.jsonl", precision_rows)

            row = {
                "user": user_label,
                "reference_model": reference_folder.name,
                "prediction_model": prediction_folder.name,
                "mode": modality_from_folder(prediction_folder),
                "reranker_model": args.reranker_model,
                "score_threshold": args.score_threshold,
                "score_transform": args.score_transform,
                "top_k": args.top_k,
                **summary,
                "reference_file": str(reference_file),
                "prediction_file": str(prediction_file),
            }
            summary_rows.append(row)
            print(
                f"{user_label} {stem}: "
                f"precision={summary['precision_rate']:.4f}, "
                f"recall={summary['recall_rate']:.4f}, "
                f"f1={summary['f1_score']:.4f}"
            )

        if summary_rows:
            summary_path = output_dir / "summary.csv"
            user_rows = [row for row in summary_rows if row["user"] == user_label]
            with summary_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(user_rows[0].keys()))
                writer.writeheader()
                writer.writerows(user_rows)
            print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
