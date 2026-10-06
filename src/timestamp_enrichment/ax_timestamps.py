import json
from pathlib import Path
from typing import Any


def load_app_events(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_accessibility(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ax_event_timestamp(app_events: list[dict], record_id: int) -> str:
    """Point timestamp for accessibility_clean id i (snapshot at app_events[i+1])."""
    idx = record_id + 1
    if idx >= len(app_events):
        return app_events[-1]["timestamp"]
    return app_events[idx]["timestamp"]


def ax_pair_range(app_events: list[dict], a: int, b: int) -> dict[str, str]:
    start = ax_event_timestamp(app_events, a)
    end_idx = b + 2
    if end_idx >= len(app_events):
        end = app_events[-1]["timestamp"]
    else:
        end = app_events[end_idx]["timestamp"]
    return {"start": start, "end": end}


def build_ax_event_timestamps(
    accessibility_path: Path,
    app_events_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    app_events = load_app_events(app_events_path)
    accessibility = load_accessibility(accessibility_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with open(output_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(accessibility):
            ts = ax_event_timestamp(app_events, i)
            precise = row.get("app_event", {}).get("timestamp_precise")
            entry = {
                "id": i,
                "timestamp": ts,
                "timestamp_precise": precise,
            }
            records.append(entry)
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return records
