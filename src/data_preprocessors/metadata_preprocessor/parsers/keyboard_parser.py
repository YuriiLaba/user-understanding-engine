from src.data_preprocessors.metadata_preprocessor.utils.io_utils import IOUtils

class KeyboardDataParser:
    """Parses and loads keyboard event data."""
    
    def __init__(self, path: str):
        self.path = path
        self.events = IOUtils.read_json(path)
        print(f"Loaded {len(self.events)} Keyboard events from {path}")