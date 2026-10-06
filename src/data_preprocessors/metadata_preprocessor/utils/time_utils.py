from datetime import datetime

class TimeUtils:
    """Time parsing and formatting utilities."""
    
    @staticmethod
    def parse_iso_to_timestamp(iso_str: str) -> float:
        """Convert ISO timestamp to Unix timestamp."""
        try:
            return datetime.fromisoformat(iso_str).timestamp()
        except ValueError:
            return 0.0
    
    @staticmethod
    def calculate_duration(start_iso: str, end_iso: str) -> float:
        """Calculate duration in seconds between two ISO timestamps."""
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        return (end - start).total_seconds()
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds as '0.00s' string."""
        return f"{seconds:.2f}s"