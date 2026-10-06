import json
from src.data_models import MetadataEvent
from src.data_preprocessors.metadata_preprocessor.utils.time_utils import TimeUtils


class IOUtils:
    """JSON I/O operations."""
    
    @staticmethod
    def read_json(path: str) -> list[dict]:
        """Read JSON file and return parsed content."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_metadata(events: list[MetadataEvent], path: str):
        """Write MetadataEvents to JSON with formatted durations."""
        data = []
        for event in events:
            event_dict = event.to_dict()
            if 'duration' in event_dict:
                event_dict['duration'] = TimeUtils.format_duration(event_dict['duration'])
            data.append(event_dict)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)