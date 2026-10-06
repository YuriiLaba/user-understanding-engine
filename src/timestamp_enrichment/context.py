from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.timestamp_enrichment.ax_timestamps import ax_pair_range, load_app_events
from src.timestamp_enrichment.frame_timestamps import build_frames_index_lookup
from src.timestamp_enrichment.metadata_timestamps import metadata_pair_range

SCREEN_MODALITIES = frozenset({"screen_ocr", "screen_pure"})


def resolve_event_json(events_dir: Path, base_name: str) -> Path:
    direct = events_dir / f"{base_name}.json"
    if direct.exists():
        return direct

    matches = sorted(events_dir.glob(f"{base_name}_*.json"))
    if matches:
        return matches[-1]
    return direct


def modality_for_pipeline(experiment_prefix: str, apply_ocr: bool = True) -> str:
    if experiment_prefix == "ax":
        return "ax"
    if experiment_prefix == "metadata":
        return "metadata"
    return "screen_ocr" if apply_ocr else "screen_pure"


def batch_timestamp_range(
    pair_indices: list[int],
    summaries: dict[int, dict[str, Any]],
) -> dict[str, str] | None:
    ranges: list[dict[str, str]] = []
    missing: list[int] = []

    for pair_idx in pair_indices:
        summary = summaries.get(pair_idx)
        if summary and "timestamp_range" in summary:
            tr = summary["timestamp_range"]
            ranges.append({"start": tr["start"], "end": tr["end"]})
        else:
            missing.append(pair_idx)

    if missing:
        print(f"[WARN] Missing pair timestamps for pair_idx: {missing}")

    if not ranges:
        return None

    return {
        "start": min(r["start"] for r in ranges),
        "end": max(r["end"] for r in ranges),
    }


@dataclass
class TimestampContext:
    modality: str
    app_events: list[dict] | None = None
    frames_lookup: dict[str, dict] | None = None

    @classmethod
    def build(
        cls,
        modality: str,
        *,
        events_dir: Path | str,
        frames_dir: Path | str | None = None,
    ) -> TimestampContext:
        events_path = Path(events_dir)

        if modality in SCREEN_MODALITIES:
            if frames_dir is None:
                raise ValueError("frames_dir is required for screen timestamp enrichment")
            frames_path = Path(frames_dir)
            if not frames_path.exists():
                raise FileNotFoundError(f"Frames directory not found: {frames_path}")
            screen_events_path = resolve_event_json(events_path, "screen_events")
            if not screen_events_path.exists():
                raise FileNotFoundError(
                    f"screen_events.json not found under {events_path}. "
                    "Required for --enrich-timestamps on screen pipelines."
                )
            lookup = build_frames_index_lookup(frames_path, screen_events_path)
            return cls(modality=modality, frames_lookup=lookup)

        app_events_path = resolve_event_json(events_path, "app_events")
        if not app_events_path.exists():
            raise FileNotFoundError(
                f"app_events.json not found under {events_path}. "
                "Required for --enrich-timestamps."
            )
        return cls(
            modality=modality,
            app_events=load_app_events(app_events_path),
        )

    def pair_timestamp_range(self, entry: dict[str, Any]) -> dict[str, str] | None:
        if self.modality == "ax":
            pair_ids = entry.get("ax_pair_ids")
            if pair_ids is None or self.app_events is None:
                return None
            a, b = pair_ids
            return ax_pair_range(self.app_events, a, b)

        if self.modality == "metadata":
            pair_ids = entry.get("metadata_pair_ids")
            if pair_ids is None or self.app_events is None:
                return None
            a, b = pair_ids
            return metadata_pair_range(self.app_events, a, b)

        if self.modality in SCREEN_MODALITIES:
            frames = entry.get("frames")
            if frames is None or self.frames_lookup is None:
                return None
            f1, f2 = frames
            row1 = self.frames_lookup.get(f1)
            row2 = self.frames_lookup.get(f2)
            if row1 is None or row2 is None:
                return None
            ts1 = row1["timestamp_iso"]
            ts2 = row2["timestamp_iso"]
            start, end = (ts1, ts2) if ts1 <= ts2 else (ts2, ts1)
            return {"start": start, "end": end}

        return None

    def attach_to_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        ts_range = self.pair_timestamp_range(entry)
        if ts_range is not None:
            entry = {**entry, "timestamp_range": ts_range}
        return entry
