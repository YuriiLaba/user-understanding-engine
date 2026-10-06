"""Cross-user semantic comparison: centroid distance matrix + 2-D scatter.

For each user, loads propositions from a fixed modality/model folder, embeds them
with a sentence-transformer, computes a per-user centroid, then:

  1. Prints a pairwise cosine-distance matrix between centroids.
  2. Saves the matrix to a CSV.
  3. Projects all propositions *and* centroids jointly to 2-D (UMAP preferred,
     t-SNE fallback) and writes an interactive Plotly HTML scatter where each
     user's cloud is one colour and centroids are rendered as large diamonds.

Usage:
    python scripts/compare_users.py
    python scripts/compare_users.py --users hlib anastasia max \\
        --modality ax --model-glob "*gpt_5.5*" --min-confidence 7
    python scripts/compare_users.py --reducer tsne --output-dir results/my_run
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proposition_io import (  # noqa: E402
    first_existing_file,
    matching_folders,
    read_structured_propositions,
)

# Okabe-Ito colorblind-safe palette
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange/amber
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

OUTPUT_DATA = PROJECT_ROOT / "output_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "compare_users"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_user_propositions(
    user_root: Path,
    model_glob: str,
    modality: str,
    min_confidence: int,
) -> list[str]:
    folders = matching_folders(user_root, [model_glob], [modality])
    if not folders:
        raise FileNotFoundError(
            f"No '{modality}' folder matching '{model_glob}' in {user_root}"
        )

    src = first_existing_file(folders[0])
    if src is None:
        raise FileNotFoundError(f"No propositions file in {folders[0]}")

    items = read_structured_propositions(src)
    texts: list[str] = []
    for item in items:
        conf = item.get("confidence")
        if conf is not None:
            try:
                if int(conf) < min_confidence:
                    continue
            except (ValueError, TypeError):
                pass
        texts.append(item["proposition"])
    return texts


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))


def compute_distance_matrix(
    centroids: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    users = list(centroids)
    n = len(users)
    matrix = np.zeros((n, n))
    for i, u1 in enumerate(users):
        for j, u2 in enumerate(users):
            if i != j:
                matrix[i, j] = _cosine_dist(centroids[u1], centroids[u2])
    return users, matrix


def print_distance_matrix(users: list[str], matrix: np.ndarray) -> None:
    col_w = max(len(u) for u in users) + 2
    print(" " * col_w + "".join(f"{u:>{col_w}}" for u in users))
    for i, u in enumerate(users):
        row = f"{u:<{col_w}}" + "".join(
            f"{'—':>{col_w}}" if i == j else f"{matrix[i, j]:>{col_w}.4f}"
            for j in range(len(users))
        )
        print(row)


def save_distance_matrix(
    users: list[str], matrix: np.ndarray, out_path: Path
) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + users)
        for i, u in enumerate(users):
            writer.writerow(
                [u]
                + ["" if i == j else f"{matrix[i, j]:.6f}" for j in range(len(users))]
            )


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    reducer: str,
    random_state: int = 42,
) -> np.ndarray:
    if reducer == "umap":
        try:
            from umap import UMAP

            n_neighbors = min(15, max(2, len(embeddings) // 10))
            print(f"Using UMAP (n_neighbors={n_neighbors})…")
            return UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=random_state,
            ).fit_transform(embeddings)
        except ImportError:
            print("umap-learn not installed — falling back to t-SNE.")
            reducer = "tsne"

    from sklearn.manifold import TSNE

    perplexity = min(30.0, max(5.0, len(embeddings) / 3))
    print(f"Using t-SNE (perplexity={perplexity:.0f})…")
    # float64 avoids overflow/NaN in sklearn's randomised PCA init on large inputs
    return TSNE(
        n_components=2,
        init="pca",
        metric="cosine",
        perplexity=perplexity,
        random_state=random_state,
    ).fit_transform(embeddings.astype(np.float64))


# ---------------------------------------------------------------------------
# Plotly scatter
# ---------------------------------------------------------------------------

def build_html_scatter(
    prop_coords: np.ndarray,
    all_labels: list[str],
    all_texts: list[str],
    centroid_coords: dict[str, np.ndarray],
    users: list[str],
    color_map: dict[str, str],
    dist_matrix: np.ndarray,
    reducer_name: str,
    title: str,
) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.72, 0.28],
        subplot_titles=["Proposition embedding space", "Centroid cosine distance"],
        horizontal_spacing=0.06,
    )

    # --- Left: point clouds + centroids ---
    for user in users:
        idx = [i for i, l in enumerate(all_labels) if l == user]
        fig.add_trace(
            go.Scatter(
                x=[float(prop_coords[i, 0]) for i in idx],
                y=[float(prop_coords[i, 1]) for i in idx],
                mode="markers",
                name=f"{user} ({len(idx)})",
                marker=dict(size=6, color=color_map[user], opacity=0.4, line=dict(width=0)),
                text=[all_texts[i] for i in idx],
                hovertemplate="<b>%{text}</b><extra>" + user + "</extra>",
                legendgroup=user,
            ),
            row=1, col=1,
        )

    for user in users:
        cx, cy = centroid_coords[user]
        fig.add_trace(
            go.Scatter(
                x=[float(cx)], y=[float(cy)],
                mode="markers+text",
                name=f"{user} ★",
                marker=dict(size=20, color=color_map[user], symbol="diamond",
                            opacity=1.0, line=dict(width=2, color="white")),
                text=[user],
                textposition="top center",
                textfont=dict(size=12, color=color_map[user]),
                hovertemplate=f"<b>{user} centroid</b><extra></extra>",
                legendgroup=user,
                showlegend=False,
            ),
            row=1, col=1,
        )

    # --- Right: distance heatmap ---
    n = len(users)
    # Round-trip symmetric matrix; blank diagonal
    z = [[None if i == j else round(float(dist_matrix[i, j]), 4) for j in range(n)] for i in range(n)]
    text_annotations = [
        [f"—" if i == j else f"{dist_matrix[i, j]:.3f}" for j in range(n)]
        for i in range(n)
    ]
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=users,
            y=users,
            colorscale="Blues",
            reversescale=False,
            showscale=True,
            colorbar=dict(title="cosine dist", len=0.5, x=1.02),
            text=text_annotations,
            texttemplate="%{text}",
            hovertemplate="<b>%{y} ↔ %{x}</b><br>distance: %{z:.4f}<extra></extra>",
            zmin=0,
            zmax=1,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(itemsizing="constant", font=dict(size=11), tracegroupgap=2),
        hovermode="closest",
        width=1400,
        height=700,
        margin=dict(l=60, r=80, t=80, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", row=1, col=1,
                     title_text=f"{reducer_name.upper()} dim 1")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", row=1, col=1,
                     title_text=f"{reducer_name.upper()} dim 2")

    return fig.to_html(full_html=True, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--propositions-root", default=str(OUTPUT_DATA),
        help="Root directory containing per-user folders (default: output_data/).",
    )
    parser.add_argument(
        "--users", nargs="+", default=None,
        help="User folder names to compare. Default: every subfolder in root.",
    )
    parser.add_argument(
        "--model-glob", default="*gpt_5.5*",
        help="Glob to select the model subfolder (default: *gpt_5.5*).",
    )
    parser.add_argument(
        "--modality", default="ax", choices=["ax", "metadata", "ocr", "screen"],
        help="Data modality to load (default: ax).",
    )
    parser.add_argument(
        "--min-confidence", type=int, default=7,
        help="Drop propositions below this confidence score 1-10 (default: 7).",
    )
    parser.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Compute device: auto | cpu | cuda | mps (default: auto).",
    )
    parser.add_argument(
        "--reducer", default="umap", choices=["umap", "tsne"],
        help="Dimensionality reduction method (default: umap, falls back to tsne).",
    )
    parser.add_argument(
        "--output-dir", default=str(RESULTS_DIR),
        help="Directory for output files (default: results/compare_users/).",
    )
    args = parser.parse_args()

    propositions_root = Path(args.propositions_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover users ---
    if args.users:
        user_names = args.users
    else:
        user_names = sorted(
            p.name for p in propositions_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    print(f"Users to compare: {user_names}")
    print(f"Modality: {args.modality}  Model glob: {args.model_glob}  "
          f"Min confidence: {args.min_confidence}\n")

    # --- Load propositions ---
    user_texts: dict[str, list[str]] = {}
    for name in user_names:
        user_root = propositions_root / name
        try:
            texts = load_user_propositions(
                user_root, args.model_glob, args.modality, args.min_confidence
            )
            user_texts[name] = texts
            print(f"  {name}: {len(texts)} propositions")
        except FileNotFoundError as exc:
            print(f"  {name}: SKIPPED — {exc}")

    if len(user_texts) < 2:
        raise SystemExit("Need at least 2 users with data to compare.")

    users = list(user_texts)
    color_map = {u: _PALETTE[i % len(_PALETTE)] for i, u in enumerate(users)}

    # --- Flatten for joint embedding ---
    all_texts: list[str] = []
    all_labels: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    for user in users:
        s = len(all_texts)
        all_texts.extend(user_texts[user])
        all_labels.extend([user] * len(user_texts[user]))
        slices[user] = (s, len(all_texts))

    # --- Embed ---
    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\nEmbedding {len(all_texts)} propositions with "
          f"'{args.embedding_model}' on {device}…")
    st_model = SentenceTransformer(args.embedding_model, device=device)
    embeddings = np.asarray(
        st_model.encode(all_texts, show_progress_bar=True, batch_size=64),
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

    # --- Centroids ---
    centroids: dict[str, np.ndarray] = {
        user: embeddings[slices[user][0]: slices[user][1]].mean(axis=0)
        for user in users
    }

    # --- Distance matrix ---
    dist_users, dist_matrix = compute_distance_matrix(centroids)
    print("\nPairwise centroid cosine distance matrix:")
    print_distance_matrix(dist_users, dist_matrix)

    matrix_path = output_dir / "centroid_distance_matrix.csv"
    save_distance_matrix(dist_users, dist_matrix, matrix_path)
    print(f"\nSaved: {matrix_path}")

    # --- Reduce: propositions + centroids jointly ---
    centroid_stack = np.stack([centroids[u] for u in users])
    combined = np.vstack([embeddings, centroid_stack])  # (N + n_users, D)

    print()
    coords_2d = reduce_dimensions(combined, args.reducer)

    prop_coords = coords_2d[: len(embeddings)]
    centroid_coords_2d: dict[str, np.ndarray] = {
        users[i]: coords_2d[len(embeddings) + i] for i in range(len(users))
    }

    # Track actual reducer used (tsne if umap fell back)
    reducer_used = args.reducer
    if args.reducer == "umap":
        try:
            import umap  # noqa: F401
        except ImportError:
            reducer_used = "tsne"

    # --- Plotly scatter ---
    title = (
        f"User semantic comparison — {args.modality} · "
        f"{args.model_glob.replace('*', '')} · {reducer_used.upper()} · "
        f"conf ≥ {args.min_confidence}"
    )
    html = build_html_scatter(
        prop_coords, all_labels, all_texts,
        centroid_coords_2d, users, color_map,
        dist_matrix, reducer_used, title,
    )

    scatter_path = output_dir / "user_comparison_scatter.html"
    scatter_path.write_text(html, encoding="utf-8")
    print(f"Saved: {scatter_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
