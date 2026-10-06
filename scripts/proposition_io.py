import csv
import json
from pathlib import Path


DEFAULT_USERS = {
    "anastasia_video_1": "anastasia_video_1",
    "anastasia_video_2": "anastasia_video_2",
    "hlib": "hlib",
    "mary": "mary (1)",
}


def read_propositions(path: Path) -> list[str]:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [
                row["proposition"].strip()
                for row in reader
                if row.get("proposition") and row["proposition"].strip()
            ]

    propositions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "proposition" in item:
                text = item["proposition"].strip()
                if text:
                    propositions.append(text)
            for proposition in item.get("propositions", []):
                text = proposition.get("proposition", "").strip()
                if text:
                    propositions.append(text)
    return propositions


def read_structured_propositions(path: Path) -> list[dict]:
    """Like read_propositions but keeps batch_id (the time axis) and confidence.

    JSONL batches carry an explicit ``batch_id``. CSV files have no time signal,
    so each row is given a sequential batch_id as a fallback ordering.
    """
    items: list[dict] = []
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                text = (row.get("proposition") or "").strip()
                if not text:
                    continue
                items.append(
                    {"proposition": text, "batch_id": idx, "confidence": row.get("confidence")}
                )
        return items

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            batch_id = record.get("batch_id")
            if "proposition" in record:
                text = record["proposition"].strip()
                if text:
                    items.append(
                        {"proposition": text, "batch_id": batch_id, "confidence": record.get("confidence")}
                    )
            for proposition in record.get("propositions", []):
                text = proposition.get("proposition", "").strip()
                if text:
                    items.append(
                        {
                            "proposition": text,
                            "batch_id": batch_id,
                            "confidence": proposition.get("confidence"),
                        }
                    )
    return items


def first_jsonl_file(folder: Path) -> Path | None:
    """Prefer the propositions.jsonl file (the only one carrying batch_id)."""
    path = folder / "propositions.jsonl"
    return path if path.exists() else None


# Observation layers: (text field, jsonl filename). Time axis for these is pair_idx.
OBSERVATION_LAYERS = {
    "summaries": ("summary", "summaries.jsonl"),
    "transcriptions": ("transcription", "transcriptions.jsonl"),
}


def layer_flat_file(folder: Path, layer: str) -> Path | None:
    """File to read for flat (text-only) use of a layer."""
    if layer == "propositions":
        return first_existing_file(folder)
    _, fname = OBSERVATION_LAYERS[layer]
    path = folder / fname
    return path if path.exists() else None


def layer_structured_file(folder: Path, layer: str) -> Path | None:
    """File to read when the time axis is needed (SAGE / moving-window)."""
    if layer == "propositions":
        return first_jsonl_file(folder) or first_existing_file(folder)
    return layer_flat_file(folder, layer)


def _read_observation_records(path: Path, field: str) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = (record.get(field) or "").strip()
            if text:
                items.append({"text": text, "time": record.get("pair_idx")})
    return items


def read_layer_flat(path: Path, layer: str) -> list[str]:
    """Text-only list for a layer (used for the ideal and non-temporal arms)."""
    if layer == "propositions":
        return read_propositions(path)
    field, _ = OBSERVATION_LAYERS[layer]
    return [item["text"] for item in _read_observation_records(path, field)]


def read_layer_structured(path: Path, layer: str) -> list[dict]:
    """Structured list normalized to {proposition, batch_id} for any layer.

    Observations use pair_idx as the batch_id (time axis).
    """
    if layer == "propositions":
        return read_structured_propositions(path)
    field, _ = OBSERVATION_LAYERS[layer]
    return [
        {"proposition": item["text"], "batch_id": item["time"], "confidence": None}
        for item in _read_observation_records(path, field)
    ]


def parse_user_folders(values: list[str] | None, propositions_root: Path) -> dict[str, Path]:
    if not values:
        return {label: propositions_root / folder for label, folder in DEFAULT_USERS.items()}

    users = {}
    for value in values:
        if "=" in value:
            label, raw_path = value.split("=", 1)
            path = Path(raw_path)
        else:
            path = Path(value)
            label = path.name

        if not path.is_absolute():
            path = propositions_root / path
        users[label] = path
    return users


def first_existing_file(folder: Path) -> Path | None:
    for name in ("unique_propositions.csv", "propositions.csv", "propositions.jsonl"):
        path = folder / name
        if path.exists():
            return path
    return None


def modality_from_folder(folder: Path) -> str:
    return folder.name.split("_", 1)[0]


def matching_folders(user_root: Path, globs: list[str], modes: list[str] | None = None) -> list[Path]:
    matches = []
    for pattern in globs:
        matches.extend(path for path in user_root.glob(pattern) if path.is_dir())

    folders = sorted(set(matches), key=lambda path: path.name)
    if modes:
        allowed = set(modes)
        folders = [folder for folder in folders if modality_from_folder(folder) in allowed]
    return folders


def first_matching_folder(user_root: Path, globs: list[str]) -> Path | None:
    folders = matching_folders(user_root, globs)
    return folders[0] if folders else None
