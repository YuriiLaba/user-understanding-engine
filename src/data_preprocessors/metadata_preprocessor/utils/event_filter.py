class EventFilter:
    """Filters events by type and time range."""
    
    @staticmethod
    def filter_by_range(
        events: list[dict],
        start_iso: str,
        end_iso: str,
        event_types: list[str]
    ) -> list[dict]:
        """Filter events within time range and matching types."""
        return [
            event for event in events
            if event.get('event_type') in event_types
            and start_iso <= event['timestamp'] < end_iso
        ]