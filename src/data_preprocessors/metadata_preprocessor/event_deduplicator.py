from itertools import groupby
from src.data_models import MetadataEvent


class EventDeduplicator:
    """Merges consecutive identical MetadataEvents."""
    
    @staticmethod
    def deduplicate(events: list[MetadataEvent]) -> list[MetadataEvent]:
        """Merge consecutive events with same trigger/app/windows."""
        def key_func(event: MetadataEvent):
            return (event.trigger_action, event.app_name, event.windows_names)
        
        merged = []
        
        for unique_key, group in groupby(events, key=key_func):
            group_list = list(group)
            base = group_list[0]
            
            # Create new merged event
            merged_event = MetadataEvent(**base.to_dict())
            merged_event.duration = sum(e.duration for e in group_list)
            
            merged.append(merged_event)
        
        # Re-index
        for idx, event in enumerate(merged):
            event.id = idx
        
        print(f"Deduplication: {len(events)} → {len(merged)} (removed {len(events) - len(merged)})")
        return merged