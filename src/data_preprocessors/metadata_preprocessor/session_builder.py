from src.data_models import Session
from src.data_preprocessors.metadata_preprocessor.utils.event_filter import EventFilter

class SessionBuilder:
    """Builds sessions from app, mouse, and keyboard events."""
    
    def __init__(self, processable_mouse_events: list[str], processable_keyboard_events: list[str]):
        self.processable_mouse_events = processable_mouse_events
        self.processable_keyboard_events = processable_keyboard_events
    
    def build_sessions(
        self,
        app_events: list[dict],
        mouse_events: list[dict],
        kb_events: list[dict]
    ) -> list[Session]:
        """Split data into sessions based on app events."""
        sessions = []
        
        for i in range(len(app_events) - 1):
            start = app_events[i]['timestamp']
            end = app_events[i + 1]['timestamp']
            
            kb_in_range = EventFilter.filter_by_range(
                kb_events, start, end, self.processable_keyboard_events
            )
            mouse_in_range = EventFilter.filter_by_range(
                mouse_events, start, end, self.processable_mouse_events
            )
            
            sessions.append(Session(
                id=len(sessions),
                start=start,
                end=end,
                app_event=app_events[i],
                kb_events=kb_in_range,
                mouse_events=mouse_in_range
            ))
        
        return sessions