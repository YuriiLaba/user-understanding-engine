"""Pair-level timestamp helpers (library-only; no file outputs)."""

from typing import Any

from src.timestamp_enrichment.ax_timestamps import ax_pair_range
from src.timestamp_enrichment.metadata_timestamps import metadata_pair_range


def ax_pair_timestamp_range(
    app_events: list[dict],
    ax_pair_ids: tuple[int, int] | list[int],
) -> dict[str, str]:
    a, b = ax_pair_ids
    return ax_pair_range(app_events, a, b)


def metadata_pair_timestamp_range(
    app_events: list[dict],
    metadata_pair_ids: tuple[int, int] | list[int],
) -> dict[str, str]:
    a, b = metadata_pair_ids
    return metadata_pair_range(app_events, a, b)


def screen_pair_timestamp_range(
    frames_lookup: dict[str, dict],
    frames: tuple[str, str] | list[str],
) -> dict[str, str] | None:
    f1, f2 = frames
    row1 = frames_lookup.get(f1)
    row2 = frames_lookup.get(f2)
    if row1 is None or row2 is None:
        return None
    ts1 = row1["timestamp_iso"]
    ts2 = row2["timestamp_iso"]
    start, end = (ts1, ts2) if ts1 <= ts2 else (ts2, ts1)
    return {"start": start, "end": end}
