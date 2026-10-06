import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proposition_io import (
    first_existing_file,
    first_matching_folder,
    matching_folders,
    modality_from_folder,
    parse_user_folders,
    read_propositions,
)


def compare_proposition_sets(
    calculator: Any,
    source_props: list[str],
    reference_props: list[str],
    semantic_jaccard_threshold: float,
    return_pairwise: bool = False,
    soft_low: float = 0.60,
    soft_high: float = 0.80,
) -> dict:
    """Embedding metrics. With return_pairwise, also returns the source x reference cosine
    matrix under "_pairwise" so callers can reuse it instead of recomputing it."""
    profile_metrics = calculator.calculate_similarity(
        {"propositions": source_props},
        {"propositions": reference_props},
    )

    if not source_props or not reference_props:
        return {
            "_pairwise": None,
            "source_to_reference_best_mean": 0.0,
            "reference_to_source_best_mean": 0.0,
            "symmetric_best_mean": 0.0,
            "semantic_jaccard": 0.0,
            "soft_jaccard": 0.0,
            "soft_precision": 0.0,
            "soft_recall": 0.0,
            "soft_f1": 0.0,
            "soft_source_regions": 0.0,
            "soft_reference_regions": 0.0,
            "profile_composite_score": profile_metrics["composite_score"],
            "profile_semantic_similarity": profile_metrics["semantic_similarity"],
            "profile_jaccard": profile_metrics["average_jaccard"],
            "profile_average_semantic": profile_metrics["average_semantic"],
            "profile_interpretation": profile_metrics["interpretation"],
        }

    source_embeddings = np.asarray(
        calculator.model.encode(source_props, show_progress_bar=False),
        dtype=np.float32,
    )
    reference_embeddings = np.asarray(
        calculator.model.encode(reference_props, show_progress_bar=False),
        dtype=np.float32,
    )

    source_embeddings = normalize_embeddings(source_embeddings)
    reference_embeddings = normalize_embeddings(reference_embeddings)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pairwise = source_embeddings.astype(np.float64) @ reference_embeddings.astype(np.float64).T
    pairwise = np.nan_to_num(pairwise, nan=0.0, posinf=0.0, neginf=0.0)
    source_to_reference = pairwise.max(axis=1)
    reference_to_source = pairwise.max(axis=0)
    symmetric_best_mean = float((np.mean(source_to_reference) + np.mean(reference_to_source)) / 2)
    semantic_jaccard = calculate_semantic_jaccard(pairwise, semantic_jaccard_threshold)
    soft = soft_coverage_metrics(source_embeddings, reference_embeddings, low=soft_low, high=soft_high)
    composite_score = (
        0.5 * profile_metrics["semantic_similarity"]
        + 0.3 * semantic_jaccard
        + 0.2 * symmetric_best_mean
    )

    return {
        **({"_pairwise": pairwise} if return_pairwise else {}),
        "source_to_reference_best_mean": float(np.mean(source_to_reference)),
        "reference_to_source_best_mean": float(np.mean(reference_to_source)),
        "symmetric_best_mean": symmetric_best_mean,
        "semantic_jaccard": semantic_jaccard,
        **soft,
        "profile_composite_score": round(composite_score, 4),
        "profile_semantic_similarity": profile_metrics["semantic_similarity"],
        "profile_jaccard": profile_metrics["average_jaccard"],
        "profile_average_semantic": profile_metrics["average_semantic"],
        "profile_interpretation": calculator._interpret_score(composite_score),
    }


def calculate_semantic_jaccard(pairwise: np.ndarray, threshold: float) -> float:
    """Greedy one-to-one semantic Jaccard using proposition-pair cosine matches."""
    candidate_indices = np.argwhere(pairwise >= threshold)
    if candidate_indices.size == 0:
        return 0.0

    candidates = sorted(
        ((float(pairwise[i, j]), int(i), int(j)) for i, j in candidate_indices),
        reverse=True,
    )
    matched_source = set()
    matched_reference = set()

    for _, source_idx, reference_idx in candidates:
        if source_idx in matched_source or reference_idx in matched_reference:
            continue
        matched_source.add(source_idx)
        matched_reference.add(reference_idx)

    intersection = len(matched_source)
    union = pairwise.shape[0] + pairwise.shape[1] - intersection
    return intersection / union if union else 0.0


def soft_coverage_metrics(
    source_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    low: float = 0.60,
    high: float = 0.80,
    probe_embeddings: np.ndarray | None = None,
) -> dict:
    """Region-based overlap of two proposition sets.

    Every proposition spans a soft region on the unit sphere: membership of a point x in the
    region of proposition p is a linear ramp of cos(x, p): 0 at or below `low`, 1 at or above
    `high`. The defaults come from scripts/calibrate_soft_metric.py run over all four sessions
    and every extraction model (~62k propositions), judged by the pipeline's cross-encoder
    (ms-marco-MiniLM-L-6-v2): cos 0.60 is where a pair becomes more likely the same fact than
    not (pooled P(match)=0.5 at 0.58) and is the 90th percentile of cosine between unrelated
    propositions about one user (0.61); at cos 0.80 every session has P(match) >= 0.99. The
    ramp has compact support, so far-away propositions add exactly nothing to a region and the
    density weights below stay meaningful. A set covers
    x with the max membership over its propositions, so the set is the union of its regions.
    Overlap is a fuzzy Jaccard, sum(min(mu_S, mu_R)) / sum(max(mu_S, mu_R)), evaluated at probe
    points. Probes default to the propositions of both sets; each probe is weighted by
    1 / (soft number of propositions that cover it), so ten paraphrases of one fact weigh the
    same as one region and cannot inflate either side. Pass a fixed external `probe_embeddings`
    (e.g. the pool of all propositions for a user) to score every run on one common yardstick;
    the weighting is then skipped.
    """
    if source_embeddings.size == 0 or reference_embeddings.size == 0:
        return {"soft_jaccard": 0.0, "soft_precision": 0.0, "soft_recall": 0.0, "soft_f1": 0.0,
                "soft_source_regions": 0.0, "soft_reference_regions": 0.0}

    def membership(probes: np.ndarray, centers: np.ndarray) -> np.ndarray:
        cos = probes.astype(np.float64) @ centers.astype(np.float64).T
        return np.clip((cos - low) / (high - low), 0.0, 1.0)  # probes x centers

    S, R = source_embeddings, reference_embeddings
    if probe_embeddings is None:
        probes = np.vstack([S, R])
        m_s, m_r = membership(probes, S), membership(probes, R)
        density = m_s.sum(axis=1) + m_r.sum(axis=1)  # soft count of regions covering each probe
        w = 1.0 / np.maximum(density, 1e-9)
    else:
        probes = probe_embeddings
        m_s, m_r = membership(probes, S), membership(probes, R)
        w = np.ones(len(probes))

    mu_s, mu_r = m_s.max(axis=1), m_r.max(axis=1)
    inter = float(np.sum(w * np.minimum(mu_s, mu_r)))
    union = float(np.sum(w * np.maximum(mu_s, mu_r)))
    vol_s = float(np.sum(w * mu_s))
    vol_r = float(np.sum(w * mu_r))
    precision = inter / vol_s if vol_s else 0.0   # share of source volume inside reference
    recall = inter / vol_r if vol_r else 0.0      # share of reference volume inside source
    return {
        "soft_jaccard": inter / union if union else 0.0,
        "soft_precision": precision,
        "soft_recall": recall,
        # harmonic mean = 2*inter / (vol_s + vol_r), i.e. the soft Dice coefficient; it is a
        # monotone function of soft_jaccard (F1 = 2J / (1 + J)) so both rank runs identically.
        "soft_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "soft_source_regions": vol_s,   # effective number of distinct regions in source
        "soft_reference_regions": vol_r,
    }


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(
        embeddings.astype(np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    clean = np.clip(clean, -1e6, 1e6)
    norms = np.linalg.norm(clean, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = clean / norms
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_rows(args: argparse.Namespace) -> list[dict]:
    from src.user_profile_similarity import UserProfileSimilarity

    calculator = UserProfileSimilarity(model_name=args.embedding_model, device=args.device)
    root = Path(args.propositions_root)
    rows = []
    users = parse_user_folders(args.user_folders, root)

    for user_name, user_root in users.items():
        baseline_folder = first_matching_folder(user_root, args.source_model_globs)
        baseline_file = first_existing_file(baseline_folder) if baseline_folder else None
        qwen_folders = matching_folders(user_root, args.reference_model_globs, args.modes)

        if not qwen_folders:
            rows.append(
                {
                    "user": user_name,
                    "mode": "",
                    "source_model": baseline_folder.name if baseline_folder else "",
                    "reference_model": "",
                    "source_file": str(baseline_file) if baseline_file else "",
                    "reference_file": "",
                    "source_count": 0,
                    "reference_count": 0,
                    "status": "missing_qwen_input",
                }
            )
            continue

        for qwen_folder in qwen_folders:
            qwen_file = first_existing_file(qwen_folder) if qwen_folder else None

            row = {
                "user": user_name,
                "mode": modality_from_folder(qwen_folder),
                "source_model": baseline_folder.name if baseline_folder else "",
                "reference_model": qwen_folder.name if qwen_folder else "",
                "source_file": str(baseline_file) if baseline_file else "",
                "reference_file": str(qwen_file) if qwen_file else "",
                "source_count": 0,
                "reference_count": 0,
                "status": "ok",
            }

            if not baseline_file or not qwen_file:
                row["status"] = "missing_input"
                rows.append(row)
                continue

            source_props = read_propositions(baseline_file)
            reference_props = read_propositions(qwen_file)
            row["source_count"] = len(source_props)
            row["reference_count"] = len(reference_props)
            row.update(
                compare_proposition_sets(
                    calculator,
                    source_props,
                    reference_props,
                    args.semantic_jaccard_threshold,
                    soft_low=args.soft_low,
                    soft_high=args.soft_high,
                )
            )
            rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare each user's screen GPT-5.1 propositions against Qwen3 proposition sets with embeddings."
    )
    parser.add_argument("--propositions-root", default="propositions")
    parser.add_argument(
        "--user-folders",
        nargs="+",
        default=None,
        help=(
            "User proposition folders to compare. Values can be folder names relative to "
            "--propositions-root, paths, or label=path. Default: built-in Anastasia/Hlib/Mary folders."
        ),
    )
    parser.add_argument(
        "--source-model-globs",
        nargs="+",
        default=["screen_gpt*5.1", "screen_gpt_5.1", "screen_gpt-5.1"],
        help="Glob(s) inside each user folder for the source/baseline model folder.",
    )
    parser.add_argument(
        "--reference-model-globs",
        nargs="+",
        default=["*_qwen3*"],
        help="Glob(s) inside each user folder for reference model folders.",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the embedding model. Use auto to prefer CUDA, then MPS, then CPU. Default: auto",
    )
    parser.add_argument(
        "--semantic-jaccard-threshold",
        type=float,
        default=0.75,
        help="Cosine threshold for considering two propositions a semantic match. Default: 0.75",
    )
    parser.add_argument("--soft-low", type=float, default=0.60,
                        help="Cosine where soft-region membership starts (see scripts/calibrate_soft_metric.py). Default: 0.60")
    parser.add_argument("--soft-high", type=float, default=0.80,
                        help="Cosine where soft-region membership reaches 1. Default: 0.80")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Optional modality filter, e.g. --modes screen metadata. Default: all Qwen3 modalities found.",
    )
    parser.add_argument(
        "--output",
        default="propositions/qwen_modalities_vs_screen_gpt_5.1_embedding_comparisons.csv",
    )
    args = parser.parse_args()

    rows = build_rows(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "user",
        "mode",
        "source_model",
        "reference_model",
        "source_count",
        "reference_count",
        "source_to_reference_best_mean",
        "reference_to_source_best_mean",
        "symmetric_best_mean",
        "semantic_jaccard",
        "soft_jaccard",
        "soft_precision",
        "soft_recall",
        "soft_f1",
        "soft_source_regions",
        "soft_reference_regions",
        "profile_composite_score",
        "profile_semantic_similarity",
        "profile_jaccard",
        "profile_average_semantic",
        "profile_interpretation",
        "status",
        "source_file",
        "reference_file",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row['user']} {row['reference_model']}: "
                f"symmetric_best_mean={row['symmetric_best_mean']:.4f}, "
                f"semantic_jaccard={row['semantic_jaccard']:.4f}, "
                f"soft_jaccard={row['soft_jaccard']:.4f}, "
                f"counts={row['source_count']}/{row['reference_count']}"
            )
        else:
            print(f"{row['user']} {row['mode']}: {row['status']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
