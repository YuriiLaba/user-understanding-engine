import json
from pathlib import Path
from typing import Any

from src.timestamp_enrichment.ax_timestamps import load_app_events


def metadata_event_range(app_events: list[dict], record_id: int) -> dict[str, str]:
    """Session window for metadata_clean id i."""
    start = app_events[record_id]["timestamp"]
    end_idx = record_id + 1
    if end_idx >= len(app_events):
        end = app_events[-1]["timestamp"]
    else:
        end = app_events[end_idx]["timestamp"]
    return {"start": start, "end": end}


def metadata_pair_range(app_events: list[dict], a: int, b: int) -> dict[str, str]:
    start = app_events[a]["timestamp"]
    end_idx = b + 1
    if end_idx >= len(app_events):
        end = app_events[-1]["timestamp"]
    else:
        end = app_events[end_idx]["timestamp"]
    return {"start": start, "end": end}


def build_metadata_event_timestamps(
    app_events_path: Path,
    output_path: Path,
    num_sessions: int | None = None,
) -> list[dict[str, Any]]:
    app_events = load_app_events(app_events_path)
    n = num_sessions if num_sessions is not None else len(app_events) - 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with open(output_path, "w", encoding="utf-8") as out:
        for i in range(n):
            window = metadata_event_range(app_events, i)
            entry = {"id": i, **window}
            records.append(entry)
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return records
