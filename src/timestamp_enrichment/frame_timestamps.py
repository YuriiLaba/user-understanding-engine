import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


FRAME_FILENAME_RE = re.compile(r"frame_(\d+)_idx_(\d+)\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class VideoSegment:
    start_idx: int
    end_idx: int
    start_time: datetime
    video_source: str
    fps: float


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def load_video_segments(screen_events_path: Path) -> list[VideoSegment]:
    with open(screen_events_path, encoding="utf-8") as f:
        events = json.load(f)

    segments: list[VideoSegment] = []
    cum = 0
    for event in events:
        data = event["data"]
        fps = data["frames_written"] / data["duration"]
        start_idx = cum + 1
        end_idx = cum + data["frames_written"]
        avi_name = Path(data["filename"]).name
        mp4_name = avi_name.replace(".avi", ".mp4")
        segments.append(
            VideoSegment(
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=parse_iso(data["start_time"]),
                video_source=mp4_name,
                fps=fps,
            )
        )
        cum += data["frames_written"]
    return segments


def total_frames_written(segments: list[VideoSegment]) -> int:
    return segments[-1].end_idx if segments else 0


def resolve_frame_timestamp(
    global_idx: int, segments: list[VideoSegment]
) -> tuple[datetime, str, float]:
    for seg in segments:
        if seg.start_idx <= global_idx <= seg.end_idx:
            offset_sec = (global_idx - seg.start_idx) / seg.fps
            ts = seg.start_time + timedelta(seconds=offset_sec)
            return ts, seg.video_source, offset_sec
    raise ValueError(f"global_frame_idx {global_idx} outside segment ranges")


def build_frames_index(
    frames_dir: Path,
    screen_events_path: Path,
    output_path: Path,
) -> list[dict]:
    segments = load_video_segments(screen_events_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(
        p.name
        for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    rows: list[dict] = []
    fieldnames = [
        "frame_id",
        "filename",
        "video_source",
        "global_frame_idx",
        "video_offset_sec",
        "timestamp_iso",
        "timestamp_unix",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filename in frame_files:
            match = FRAME_FILENAME_RE.match(filename)
            if not match:
                continue
            frame_id = int(match.group(1))
            global_idx = int(match.group(2))
            ts, video_source, offset_sec = resolve_frame_timestamp(global_idx, segments)
            row = {
                "frame_id": frame_id,
                "filename": filename,
                "video_source": video_source,
                "global_frame_idx": global_idx,
                "video_offset_sec": round(offset_sec, 6),
                "timestamp_iso": ts.isoformat(),
                "timestamp_unix": ts.timestamp(),
            }
            rows.append(row)
            writer.writerow(row)

    return rows


def load_frames_index_csv(path: Path) -> dict[str, dict]:
    """Map filename -> row from frames_index.csv."""
    lookup: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["filename"]] = row
    return lookup


def build_frames_index_lookup(
    frames_dir: Path,
    screen_events_path: Path,
) -> dict[str, dict]:
    """Build in-memory filename -> row map (no CSV written)."""
    segments = load_video_segments(screen_events_path)
    lookup: dict[str, dict] = {}

    for path in frames_dir.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        match = FRAME_FILENAME_RE.match(path.name)
        if not match:
            continue
        frame_id = int(match.group(1))
        global_idx = int(match.group(2))
        ts, video_source, offset_sec = resolve_frame_timestamp(global_idx, segments)
        lookup[path.name] = {
            "frame_id": frame_id,
            "filename": path.name,
            "video_source": video_source,
            "global_frame_idx": global_idx,
            "video_offset_sec": round(offset_sec, 6),
            "timestamp_iso": ts.isoformat(),
            "timestamp_unix": ts.timestamp(),
        }

    return lookup
