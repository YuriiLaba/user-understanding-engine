"""Calibrate the `low`/`high` cosine bounds of the soft-coverage metric against the reranker.

The soft metric (scripts/compare_proposition_embeddings.py::soft_coverage_metrics) treats two
propositions as sharing a region with membership that ramps linearly from 0 at cosine `low` to
1 at cosine `high`. This script derives those bounds from data rather than by hand:

  1. pool the propositions of every user/model folder found under --propositions-root,
  2. sample proposition pairs stratified by embedding cosine,
  3. ask the pipeline's cross-encoder judge whether each pair states the same fact,
  4. report P(match | cosine) per user and pooled, plus the cosine distribution of random pairs
     (the background: pairs about the same user that share no fact still have cosine ~0.45).

Recommended bounds:
  low  = the larger of (a) the pooled cosine where P(match) reaches --low-level (default 0.50,
         i.e. a pair is more likely the same fact than not) and (b) the --background-quantile
         (default 0.90) of random-pair cosine, so unrelated propositions get zero membership;
         rounded to the nearest 0.05
  high = pooled cosine where P(match) reaches --high-level (default 0.99); rounded to the nearest 0.05

On the four sessions in new_data (hlib, anastasia, anastasia_video, pavlo; ~62k propositions
from every extraction model) this gives low=0.60, high=0.80, which are the defaults of
soft_coverage_metrics. Re-run this script when the embedding model, the reranker or the
domain changes.

Run: .venv/bin/python scripts/calibrate_soft_metric.py --propositions-root new_data
     .venv/bin/python scripts/calibrate_soft_metric.py --users hlib pavlo --models qwen3_8b gpt_5.5
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_proposition_embeddings import normalize_embeddings  # noqa: E402
from scripts.proposition_io import first_existing_file, read_propositions  # noqa: E402

MODALITIES = ("ax", "metadata", "ocr", "screen")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_user_propositions(user_root: Path, models: list[str] | None) -> list[str]:
    props: list[str] = []
    for folder in sorted(user_root.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split("_", 1)
        if len(parts) != 2 or parts[0] not in MODALITIES:
            continue
        if models and parts[1] not in models:
            continue
        f = first_existing_file(folder)
        if f:
            props += read_propositions(f)
    return list(dict.fromkeys(props))


def stratified_pairs(E: np.ndarray, bins: np.ndarray, per_bin: int, anchors: int, rng) -> list[tuple[int, int, float]]:
    n = len(E)
    idx = rng.choice(n, size=min(anchors, n), replace=False)
    C = E[idx] @ E.T
    not_self = idx[:, None] != np.arange(n)[None, :]
    pairs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        cand = np.argwhere((C >= lo) & (C < hi) & not_self)
        if len(cand) == 0:
            continue
        take = cand[rng.choice(len(cand), size=min(per_bin, len(cand)), replace=False)]
        pairs += [(int(idx[i]), int(j), float(C[i, j])) for i, j in take]
    return pairs


def crossing(p: np.ndarray, centres: np.ndarray, level: float) -> float:
    """Cosine at which the (monotone-ish) match curve first reaches `level`."""
    ok = ~np.isnan(p)
    p, c = p[ok], centres[ok]
    p = np.maximum.accumulate(p)  # enforce monotone so interpolation is well defined
    return float(np.interp(level, p, c))


def round_up(x: float, step: float = 0.05) -> float:
    return math.ceil(x / step - 1e-9) * step


def round_nearest(x: float, step: float = 0.05) -> float:
    return round(x / step) * step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--propositions-root", default="new_data")
    ap.add_argument("--users", nargs="+", default=None, help="user folders to use (default: every folder found)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="model suffixes to include, e.g. gpt_5.5 qwen3_8b (default: every model)")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--reranker-threshold", type=float, default=0.5, help="same as the judge's --score-threshold")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bin-width", type=float, default=0.05)
    ap.add_argument("--pairs-per-bin", type=int, default=120, help="per user")
    ap.add_argument("--anchors", type=int, default=1500, help="anchor propositions per user to draw pairs from")
    ap.add_argument("--random-pairs", type=int, default=3000, help="per user, for the background distribution")
    ap.add_argument("--low-level", type=float, default=0.50)
    ap.add_argument("--high-level", type=float, default=0.99)
    ap.add_argument("--background-quantile", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the recommendation and tables as JSON")
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer

    rng = np.random.default_rng(args.seed)
    embedder = SentenceTransformer(args.embedding_model, device=args.device)
    reranker = CrossEncoder(args.reranker_model, device=args.device)

    root = Path(args.propositions_root)
    users = args.users or sorted(d.name for d in root.iterdir() if d.is_dir())
    bins = np.arange(0.30, 1.0 + 1e-9, args.bin_width)
    centres = (bins[:-1] + bins[1:]) / 2

    cos_all, match_all, user_all, background = [], [], [], {}
    for u in users:
        props = load_user_propositions(root / u, args.models)
        if len(props) < 50:
            print(f"[{u}] only {len(props)} propositions, skipped", file=sys.stderr)
            continue
        E = normalize_embeddings(np.asarray(embedder.encode(props, show_progress_bar=False, batch_size=256))).astype(np.float64)
        pairs = stratified_pairs(E, bins, args.pairs_per_bin, args.anchors, rng)
        texts = [(props[i], props[j]) for i, j, _ in pairs]
        s_ab = reranker.predict(texts, show_progress_bar=False, batch_size=128)
        s_ba = reranker.predict([(b, a) for a, b in texts], show_progress_bar=False, batch_size=128)
        score = np.maximum(sigmoid(s_ab), sigmoid(s_ba))
        cos_all += [c for *_, c in pairs]
        match_all += list(score >= args.reranker_threshold)
        user_all += [u] * len(pairs)
        ri, rj = rng.choice(len(E), args.random_pairs), rng.choice(len(E), args.random_pairs)
        rc = np.sum(E[ri] * E[rj], axis=1)
        background[u] = {"n_props": len(props), "mean": float(rc.mean()),
                         "q": {str(q): float(np.quantile(rc, q)) for q in (0.5, 0.9, 0.95, 0.99)},
                         "_samples": rc}
        print(f"[{u}] {len(props)} propositions, {len(pairs)} judged pairs", file=sys.stderr)

    cos_all, match_all, user_all = np.array(cos_all), np.array(match_all, dtype=float), np.array(user_all)
    used = [u for u in users if u in background]

    def curve(mask_user):
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = mask_user & (cos_all >= lo) & (cos_all < hi)
            out.append(float(match_all[mk].mean()) if mk.sum() else np.nan)
        return np.array(out)

    curves = {u: curve(user_all == u) for u in used}
    curves["pooled"] = curve(np.ones_like(cos_all, dtype=bool))

    print("\nrandom-pair cosine background (pairs about the same user, no shared fact)")
    print(f"{'user':18s}{'props':>7s}{'mean':>7s}{'p90':>7s}{'p95':>7s}{'p99':>7s}")
    for u in used:
        b = background[u]
        print(f"{u:18s}{b['n_props']:7d}{b['mean']:7.3f}{b['q']['0.9']:7.3f}{b['q']['0.95']:7.3f}{b['q']['0.99']:7.3f}")
    pooled_bg = np.concatenate([background[u]["_samples"] for u in used])
    bg_guard = float(np.quantile(pooled_bg, args.background_quantile))
    print(f"{'pooled':18s}{'':7s}{pooled_bg.mean():7.3f}{np.quantile(pooled_bg, .9):7.3f}"
          f"{np.quantile(pooled_bg, .95):7.3f}{np.quantile(pooled_bg, .99):7.3f}")

    print(f"\nP(reranker match >= {args.reranker_threshold}) per cosine bin")
    names = used + ["pooled"]
    print(f"{'cosine':12s}" + "".join(f"{n[:11]:>12s}" for n in names))
    for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        row = f"{lo:.2f}-{hi:.2f}  "
        for n in names:
            v = curves[n][k]
            row += f"{v:12.2f}" if not np.isnan(v) else f"{'-':>12s}"
        print(row)

    print("\ncosine at which P(match) reaches a level")
    levels = (args.low_level, 0.5, args.high_level)
    print(f"{'user':18s}" + "".join(f"{l:>8.2f}" for l in levels))
    cross = {}
    for n in names:
        cross[n] = [crossing(curves[n], centres, l) for l in levels]
        print(f"{n:18s}" + "".join(f"{c:8.3f}" for c in cross[n]))

    low_raw = max(cross["pooled"][0], bg_guard)
    low, high = round_nearest(low_raw), round_nearest(cross["pooled"][2])
    print(f"\nrecommendation: low={low:.2f}  high={high:.2f}")
    print(f"  low : max(P(match)={args.low_level:.2f} crossing {cross['pooled'][0]:.3f}, "
          f"background q{args.background_quantile:.2f} {bg_guard:.3f}) -> nearest 0.05")
    print(f"  high: P(match)={args.high_level:.2f} crossing {cross['pooled'][2]:.3f} -> nearest 0.05")
    print(f"  per-user P(match) at low={low:.2f}: " +
          ", ".join(f"{u}={np.interp(low, centres, np.nan_to_num(curves[u], nan=0.0)):.2f}" for u in used))
    print(f"  per-user P(match) at high={high:.2f}: " +
          ", ".join(f"{u}={np.interp(high, centres, np.nan_to_num(curves[u], nan=1.0)):.2f}" for u in used))

    if args.out:
        payload = {
            "low": low, "high": high, "users": used, "models": args.models,
            "reranker_model": args.reranker_model, "embedding_model": args.embedding_model,
            "bins": [float(b) for b in bins],
            "p_match": {n: [None if np.isnan(v) else float(v) for v in c] for n, c in curves.items()},
            "crossings": {n: dict(zip([str(l) for l in levels], c)) for n, c in cross.items()},
            "background": {u: {k: v for k, v in b.items() if k != "_samples"} for u, b in background.items()},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
