"""Matplotlib 3D scatter of the ideal (gpt_5.5) memory, colored by modality.

Reuses the 3D t-SNE coords produced by plot_ideal_embeddings.py --json-out, so it
does not re-embed or re-run t-SNE.

Requires matplotlib:  uv pip install matplotlib

    python scripts/plot_ideal_embeddings_mpl.py \
        --json results/anastasia/ideal_embeddings_3d.json \
        --output results/anastasia/ideal_embeddings_mpl_3d.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODALITIES = ["ax", "metadata", "ocr", "screen"]
COLORS = {"ax": "#0072B2", "metadata": "#E69F00", "ocr": "#009E73", "screen": "#D55E00"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default="results/anastasia/ideal_embeddings_3d.json")
    parser.add_argument("--output", default="results/anastasia/ideal_embeddings_mpl_3d.png")
    parser.add_argument("--point-size", type=float, default=16.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--views",
        nargs="+",
        default=["20,-60", "20,30", "20,120", "70,-90"],
        help="Camera angles as elev,azim pairs. One subplot per view. Default: 4 sides.",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    pts = data["points"]
    counts = data.get("counts", {})
    views = [tuple(float(v) for v in pair.split(",")) for pair in args.views]

    by_mod = {m: [p for p in pts if p["m"] == m] for m in MODALITIES}

    n = len(views)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(7.5 * cols, 6.2 * rows))

    handles = None
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_facecolor("white")
        for modality in MODALITIES:
            sel = by_mod[modality]
            if not sel:
                continue
            ax.scatter(
                [p["x"] for p in sel], [p["y"] for p in sel], [p["z"] for p in sel],
                s=args.point_size, c=COLORS[modality], alpha=0.9,
                edgecolors="white", linewidths=0.25, depthshade=False,
                label=f"{modality} ({counts.get(modality, len(sel))})",
            )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev:g}, azim={azim:g}", fontsize=10, color="#566462")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.grid(True, alpha=0.25)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.suptitle("gpt_5.5 ideal memory — proposition embeddings (t-SNE 3D)", fontsize=15, y=0.98)
    fig.legend(handles, labels, loc="upper right", markerscale=2.2, framealpha=0.95, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output} ({n} views)")


if __name__ == "__main__":
    main()
