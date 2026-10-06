"""t-SNE scatter of the ideal (gpt_5.5) memory, colored by modality.

Embeds each modality's propositions of the ideal model, compresses to 2D (or 3D
projected to 2D) with t-SNE, and renders a PNG with one color per modality.
Rendered with Pillow (no matplotlib dependency).

    python scripts/plot_ideal_embeddings.py --user anastasia --dim 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proposition_io import layer_flat_file, matching_folders, read_layer_flat  # noqa: E402

MODALITIES = ["ax", "metadata", "ocr", "screen"]
# Okabe-Ito colorblind-safe categorical hues (validated).
COLORS = {
    "ax": (0, 114, 178),      # blue
    "metadata": (230, 159, 0),  # orange
    "ocr": (0, 158, 115),     # green
    "screen": (213, 94, 0),   # vermillion
}


def get_font(size: int):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def load_points(user_root: Path, model_globs, modalities, layer):
    texts, labels = [], []
    counts = {}
    for modality in modalities:
        folders = matching_folders(user_root, model_globs, [modality])
        if not folders:
            counts[modality] = 0
            continue
        src = layer_flat_file(folders[0], layer)
        items = read_layer_flat(src, layer) if src else []
        texts.extend(items)
        labels.extend([modality] * len(items))
        counts[modality] = len(items)
    return texts, labels, counts


def project_3d_to_2d(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate 3D t-SNE coords and project to 2D; return (xy, depth) for shading."""
    ax, ay = np.deg2rad(25), np.deg2rad(35)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rotated = coords @ rx.T @ ry.T
    return rotated[:, :2], rotated[:, 2]


def render(coords, labels, depth, counts, title, subtitle, out_path, dim):
    S = 2  # supersample for anti-aliasing
    W, H = 1500 * S, 1040 * S
    pad = 70 * S
    plot_w, plot_h = W - 2 * pad, H - 2 * pad - 60 * S

    bg = (252, 252, 251)
    ink = (20, 32, 30)
    muted = (86, 100, 98)
    hair = (220, 227, 225)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    f_title = get_font(30 * S)
    f_sub = get_font(15 * S)
    f_leg = get_font(16 * S)

    d.text((pad, 30 * S), title, font=f_title, fill=ink)
    d.text((pad, 72 * S), subtitle, font=f_sub, fill=muted)

    x, y = coords[:, 0], coords[:, 1]
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    top = 120 * S

    def to_px(px, py):
        sx = pad + (px - x0) / (x1 - x0 + 1e-9) * plot_w
        sy = top + (1 - (py - y0) / (y1 - y0 + 1e-9)) * plot_h
        return sx, sy

    # subtle plot frame
    d.rectangle([pad, top, pad + plot_w, top + plot_h], outline=hair, width=1 * S)

    # depth ordering for 3D (draw far first)
    order = np.argsort(-depth) if depth is not None else range(len(coords))
    base_r = (5 if dim == 2 else 6) * S
    for idx in order:
        sx, sy = to_px(x[idx], y[idx])
        r = base_r
        col = COLORS[labels[idx]]
        if depth is not None:
            t = (depth[idx] - depth.min()) / (np.ptp(depth) + 1e-9)  # 0 far .. 1 near
            r = int(base_r * (0.6 + 0.6 * t))
            col = tuple(int(c * (0.55 + 0.45 * t) + bg[i] * (1 - (0.55 + 0.45 * t))) for i, c in enumerate(col))
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=col, outline=bg, width=1 * S)

    # legend (top-right)
    lx = pad + plot_w - 230 * S
    ly = top + 16 * S
    d.rectangle([lx - 16 * S, ly - 12 * S, pad + plot_w - 12 * S, ly + len(MODALITIES) * 30 * S + 4 * S],
                fill=(255, 255, 255), outline=hair, width=1 * S)
    for modality in MODALITIES:
        if counts.get(modality, 0) == 0:
            continue
        col = COLORS[modality]
        d.ellipse([lx, ly + 2 * S, lx + 18 * S, ly + 20 * S], fill=col, outline=bg, width=1 * S)
        d.text((lx + 28 * S, ly), f"{modality}  ({counts[modality]})", font=f_leg, fill=ink)
        ly += 30 * S

    img = img.resize((W // S, H // S), Image.LANCZOS)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--propositions-root", default="new_data")
    parser.add_argument("--user", default="anastasia")
    parser.add_argument("--model-globs", nargs="+", default=["*gpt_5.5*", "*gpt-5.5*", "*gpt*5.5*"])
    parser.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES)
    parser.add_argument("--input-layer", default="propositions", choices=["propositions", "summaries", "transcriptions"])
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-out", default=None, help="Also dump coords+labels JSON (for the interactive viewer).")
    args = parser.parse_args()

    user_root = Path(args.propositions_root) / args.user
    texts, labels, counts = load_points(user_root, args.model_globs, args.modalities, args.input_layer)
    print(f"Loaded {len(texts)} items: " + ", ".join(f"{m}={counts.get(m,0)}" for m in args.modalities))
    if len(texts) < 5:
        raise SystemExit("Not enough points to plot.")

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Embedding with {args.embedding_model} on {device}...")
    model = SentenceTransformer(args.embedding_model, device=device)
    emb = np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    perplexity = min(args.perplexity, max(5, len(texts) / 3))
    print(f"Running t-SNE (dim={args.dim}, perplexity={perplexity:.0f})...")
    tsne = TSNE(n_components=args.dim, init="pca", metric="euclidean",
                perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(emb)

    if args.json_out:
        import json

        pts = [
            {"x": round(float(c[0]), 3), "y": round(float(c[1]), 3),
             "z": round(float(c[2]), 3) if args.dim == 3 else 0.0, "m": labels[i]}
            for i, c in enumerate(coords)
        ]
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps({"points": pts, "counts": counts, "dim": args.dim}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved: {args.json_out}")

    depth = None
    if args.dim == 3:
        coords, depth = project_3d_to_2d(coords)

    labels_arr = labels
    out = args.output or f"results/{args.user}/ideal_embeddings_tsne_{args.dim}d.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    title = f"gpt_5.5 ideal memory — proposition embeddings (t-SNE {args.dim}D)"
    subtitle = f"user: {args.user}  ·  {len(texts)} propositions  ·  colored by modality"
    render(np.asarray(coords), labels_arr, depth, counts, title, subtitle, out, args.dim)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
