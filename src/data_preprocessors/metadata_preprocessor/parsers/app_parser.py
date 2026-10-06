from src.data_preprocessors.metadata_preprocessor.utils.io_utils import IOUtils

class AppDataParser:
    """Parses and loads application event data."""
    
    def __init__(self, path: str):
        self.path = path
        self.events = IOUtils.read_json(path)
        print(f"Loaded {len(self.events)} App events from {path}")
    
    @staticmethod
    def extract_window_titles(app_event: dict) -> str:
        """Extract window information from app event."""
        data = app_event['data']
        window_titles = data.get('window_titles')
        
        if window_titles:
            return f"Window Titles: {window_titles}"
        
        current = data.get('current_windows')
        added = data.get('added_windows')
        removed = data.get('removed_windows')
        
        if current:
            return f"Current Windows: {current}, Added: {added}, Removed: {removed}"
        
        return "No window info"