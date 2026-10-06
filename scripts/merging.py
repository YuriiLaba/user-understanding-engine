"""Proposition merging strategies (the experimental arms).

Each merger takes a flat list of proposition strings and returns a reduced
list plus stats. This module currently implements the embeddings + reranker
baseline; sage+time and moving-window can be added alongside it later.
"""

import math
from typing import Any

import numpy as np
from tqdm import tqdm

from scripts.compare_proposition_embeddings import normalize_embeddings
from scripts.reranker_judge import transform_scores


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def merge_embeddings_reranker(
    propositions: list[str],
    embedder: Any,
    reranker: Any,
    sim_threshold: float,
    reranker_threshold: float,
    score_transform: str = "sigmoid",
    batch_size: int = 64,
    synthesize_fn: Any = None,
) -> tuple[list[str], dict]:
    """Merge near-duplicate propositions: embedding recall -> reranker confirms.

    1. Embed all propositions and take pairs with cosine >= ``sim_threshold``.
    2. Rerank those candidate pairs; keep pairs scoring >= ``reranker_threshold``
       as merge edges.
    3. Connected components become clusters. Each multi-member cluster collapses
       to a single proposition: if ``synthesize_fn`` is given it produces the
       merged text (LLM synthesis); otherwise the cluster's medoid (most central
       proposition by embedding similarity) is kept. A failing ``synthesize_fn``
       falls back to the medoid.
    """
    n = len(propositions)
    empty_stats = {
        "merge_input_count": n,
        "merge_output_count": n,
        "merge_pairs_considered": 0,
        "merge_edges": 0,
        "merge_clusters_reduced": 0,
    }
    if n <= 1:
        return list(propositions), empty_stats

    embeddings = np.asarray(embedder.encode(propositions, show_progress_bar=False), dtype=np.float32)
    embeddings = normalize_embeddings(embeddings)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = embeddings.astype(np.float64) @ embeddings.astype(np.float64).T
    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

    upper_i, upper_j = np.triu_indices(n, k=1)
    recall_mask = sims[upper_i, upper_j] >= sim_threshold
    cand_i = upper_i[recall_mask]
    cand_j = upper_j[recall_mask]
    pairs_considered = int(cand_i.size)

    edges: list[tuple[int, int]] = []
    if pairs_considered:
        pair_texts = [(propositions[i], propositions[j]) for i, j in zip(cand_i, cand_j)]
        # batch_size reaches the GPU via predict() (default would otherwise be 32).
        raw = reranker.predict(pair_texts, batch_size=batch_size, show_progress_bar=False)
        transformed = transform_scores(np.asarray(raw, dtype=float).reshape(-1), score_transform)
        for k, (i, j) in enumerate(zip(cand_i, cand_j)):
            if transformed[k] >= reranker_threshold:
                edges.append((int(i), int(j)))

    uf = _UnionFind(n)
    for i, j in edges:
        uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)

    merged: list[str] = []
    clusters_reduced = 0
    llm_calls = 0

    def medoid_text(members: list[int]) -> str:
        sub = sims[np.ix_(members, members)]
        medoid_local = int(np.argmax(sub.sum(axis=1)))
        return propositions[members[medoid_local]]

    for members in clusters.values():
        if len(members) == 1:
            merged.append(propositions[members[0]])
            continue
        clusters_reduced += 1
        cluster_texts = [propositions[m] for m in members]

        if synthesize_fn is not None:
            try:
                merged.append(synthesize_fn(cluster_texts))
                llm_calls += 1
                continue
            except Exception as exc:  # noqa: BLE001 - fall back to medoid on any LLM failure
                print(f"  synthesize_fn failed ({exc}); falling back to medoid")
        merged.append(medoid_text(members))

    stats = {
        "merge_input_count": n,
        "merge_output_count": len(merged),
        "merge_pairs_considered": pairs_considered,
        "merge_edges": len(edges),
        "merge_clusters_reduced": clusters_reduced,
        "merge_llm_calls": llm_calls,
    }
    return merged, stats


def merge_sage_time_window(
    items: list[dict],
    embedder: Any,
    novelty_threshold: float,
    time_window: int | None,
    decay_lambda: float | None = None,
    synthesize_fn: Any = None,
) -> tuple[list[str], dict]:
    """SAGE-style online novelty gate constrained to a temporal window.

    Propositions are processed in ``batch_id`` (time) order. Each incoming
    proposition is compared only against already-accepted memories whose batch
    distance is within ``time_window`` (this is what stops similar-but-temporally-
    distant propositions from being merged). Similarity may be decay-weighted by
    ``exp(-Δbatch / decay_lambda)``. If the best in-window similarity is >=
    ``novelty_threshold`` the proposition is absorbed (redundant); otherwise it is
    kept as a new memory.

    Each kept memory collapses to its earliest (first-accepted) text, or - if
    ``synthesize_fn`` is given and it absorbed others - to an LLM-synthesized text.
    """
    texts = [it["proposition"] for it in items]
    batch = [
        it["batch_id"] if it.get("batch_id") is not None else idx
        for idx, it in enumerate(items)
    ]
    n = len(texts)
    empty_stats = {
        "merge_input_count": n,
        "merge_output_count": n,
        "merge_absorbed": 0,
        "merge_time_window": time_window,
        "merge_llm_calls": 0,
    }
    if n == 0:
        return [], empty_stats

    embeddings = np.asarray(embedder.encode(texts, show_progress_bar=False), dtype=np.float32)
    embeddings = normalize_embeddings(embeddings).astype(np.float64)

    order = sorted(range(n), key=lambda i: (batch[i], i))
    accepted: list[int] = []           # indices of kept memories, in acceptance order
    clusters: list[list[int]] = []     # members absorbed into each accepted memory
    absorbed = 0

    for i in order:
        best_sim = -1.0
        best_pos = None
        for pos, a in enumerate(accepted):
            dt = abs(batch[i] - batch[a])
            if time_window is not None and dt > time_window:
                continue
            sim = float(embeddings[i] @ embeddings[a])
            if decay_lambda:
                sim *= math.exp(-dt / decay_lambda)
            if sim > best_sim:
                best_sim = sim
                best_pos = pos

        if best_pos is not None and best_sim >= novelty_threshold:
            absorbed += 1
            clusters[best_pos].append(i)
        else:
            accepted.append(i)
            clusters.append([i])

    merged: list[str] = []
    llm_calls = 0
    for pos, members in enumerate(clusters):
        anchor = accepted[pos]
        if len(members) == 1 or synthesize_fn is None:
            merged.append(texts[anchor])
            continue
        try:
            merged.append(synthesize_fn([texts[m] for m in members]))
            llm_calls += 1
        except Exception as exc:  # noqa: BLE001 - fall back to earliest text
            print(f"  synthesize_fn failed ({exc}); keeping earliest proposition")
            merged.append(texts[anchor])

    stats = {
        "merge_input_count": n,
        "merge_output_count": len(merged),
        "merge_absorbed": absorbed,
        "merge_time_window": time_window,
        "merge_llm_calls": llm_calls,
    }
    return merged, stats


def _cluster_reduce(
    texts: list[str],
    embedder: Any,
    sim_threshold: float,
    synthesize_fn: Any = None,
) -> tuple[list[str], int, list[int]]:
    """Embedding-threshold clustering of a small text set -> one text per cluster.

    Returns (texts, llm_calls, representative index per output). The representative is the
    cluster medoid; with ``synthesize_fn`` the text is rewritten but the index still points at
    the medoid, so callers can recover its metadata (time, source modality, transcription).
    """
    n = len(texts)
    if n == 0:
        return [], 0, []
    if n == 1:
        return [texts[0]], 0, [0]

    embeddings = normalize_embeddings(
        np.asarray(embedder.encode(texts, show_progress_bar=False), dtype=np.float32)
    ).astype(np.float64)
    sims = np.nan_to_num(embeddings @ embeddings.T)

    uf = _UnionFind(n)
    ui, uj = np.triu_indices(n, k=1)
    for i, j in zip(ui, uj):
        if sims[i, j] >= sim_threshold:
            uf.union(int(i), int(j))

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)

    out: list[str] = []
    reps: list[int] = []
    llm_calls = 0
    for members in clusters.values():
        if len(members) == 1:
            out.append(texts[members[0]])
            reps.append(members[0])
            continue
        sub = sims[np.ix_(members, members)]
        medoid = members[int(np.argmax(sub.sum(axis=1)))]
        reps.append(medoid)
        cluster_texts = [texts[m] for m in members]
        if synthesize_fn is not None:
            try:
                out.append(synthesize_fn(cluster_texts))
                llm_calls += 1
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"  synthesize_fn failed ({exc}); medoid fallback")
        out.append(texts[medoid])
    return out, llm_calls, reps


def merge_moving_window(
    items: list[dict],
    embedder: Any,
    window_size: int,
    sim_threshold: float,
    synthesize_fn: Any = None,
    long_term: bool = True,
    return_indices: bool = False,
    show_progress_bar: bool = False,
) -> tuple[list[str], dict] | tuple[list[str], dict, list[int]]:
    """Two-pass tumbling-window consolidation (our method).

    Pass 1 (intra-window): items are partitioned into non-overlapping windows of
    ``window_size`` time units (batch_id for propositions, pair_idx for
    observations). Near-duplicates within each window are merged into that
    window's "main memories".

    Pass 2 (cross-window): each window's mains are rolled up into a running
    long-term store in time order; a main that matches an existing memory
    (cosine >= ``sim_threshold``) is absorbed (promoting a stable trait), else it
    is appended. ``synthesize_fn`` writes merged text when given; otherwise the
    medoid / first-seen memory is kept.

    ``batch_id`` is any monotone time value: a batch counter, or epoch seconds when the
    caller wants wall-clock windows (then ``window_size`` is in seconds). ``long_term=False``
    stops after pass 1 (windowed de-duplication only). ``return_indices=True`` also returns,
    for every output text, the index into ``items`` of the item it represents.
    """
    texts = [it["proposition"] for it in items]
    times = [
        it["batch_id"] if it.get("batch_id") is not None else idx
        for idx, it in enumerate(items)
    ]
    n = len(texts)
    empty_stats = {
        "merge_input_count": n,
        "merge_windows": 0,
        "merge_pass1_count": n,
        "merge_output_count": n,
        "merge_window_size": window_size,
        "merge_llm_calls": 0,
    }
    if n == 0:
        return ([], empty_stats, []) if return_indices else ([], empty_stats)

    t_min = min(times)
    windows: dict[int, list[int]] = {}
    for idx, t in enumerate(times):
        windows.setdefault(int((t - t_min) // window_size), []).append(idx)

    # Pass 1 - consolidate within each window
    mains: list[str] = []
    main_src: list[int] = []  # item index each main represents
    llm_calls = 0
    sorted_windows = sorted(windows)
    window_iter = tqdm(sorted_windows, desc="pass-1 windows", unit="win") if show_progress_bar else sorted_windows
    for w in window_iter:
        window_texts = [texts[i] for i in windows[w]]
        reduced, calls, reps = _cluster_reduce(window_texts, embedder, sim_threshold, synthesize_fn)
        mains.extend(reduced)
        main_src.extend(windows[w][r] for r in reps)
        llm_calls += calls
    pass1_count = len(mains)

    if not long_term:
        stats = {
            "merge_input_count": n,
            "merge_windows": len(windows),
            "merge_pass1_count": pass1_count,
            "merge_output_count": pass1_count,
            "merge_window_size": window_size,
            "merge_llm_calls": llm_calls,
        }
        return (mains, stats, main_src) if return_indices else (mains, stats)

    # Pass 2 - roll window mains up into the long-term store
    mains_emb = normalize_embeddings(
        np.asarray(embedder.encode(mains, show_progress_bar=show_progress_bar), dtype=np.float32)
    ).astype(np.float64)

    store: list[str] = []
    store_src: list[int] = []
    lt_emb: list[np.ndarray] = []
    for k, main in enumerate(mains):
        vec = mains_emb[k]
        if lt_emb:
            sims = np.asarray(lt_emb) @ vec
            best = int(np.argmax(sims))
            if sims[best] >= sim_threshold:
                if synthesize_fn is not None:
                    try:
                        new_text = synthesize_fn([store[best], main])
                        store[best] = new_text
                        lt_emb[best] = normalize_embeddings(
                            np.asarray(embedder.encode([new_text], show_progress_bar=False), dtype=np.float32)
                        ).astype(np.float64)[0]
                        llm_calls += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"  synthesize_fn failed ({exc}); keeping existing memory")
                continue
        store.append(main)
        store_src.append(main_src[k])
        lt_emb.append(vec)

    stats = {
        "merge_input_count": n,
        "merge_windows": len(windows),
        "merge_pass1_count": pass1_count,
        "merge_output_count": len(store),
        "merge_window_size": window_size,
        "merge_llm_calls": llm_calls,
    }
    return (store, stats, store_src) if return_indices else (store, stats)
