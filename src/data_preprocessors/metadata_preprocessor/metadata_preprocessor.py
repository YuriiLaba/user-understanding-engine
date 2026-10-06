from src.data_preprocessors.metadata_preprocessor.parsers.app_parser import AppDataParser
from src.data_preprocessors.metadata_preprocessor.parsers.mouse_parser import MouseDataParser
from src.data_preprocessors.metadata_preprocessor.parsers.keyboard_parser import KeyboardDataParser
from src.data_preprocessors.metadata_preprocessor.session_builder import SessionBuilder
from src.data_preprocessors.metadata_preprocessor.event_processor import EventProcessor
from src.data_preprocessors.metadata_preprocessor.utils.io_utils import IOUtils
from src.data_preprocessors.metadata_preprocessor.event_deduplicator import EventDeduplicator

class MetadataPreprocessor:
    def __init__(
        self, processable_mouse_events: list[str] = ['mouse_click', 'drag'],
          processable_keyboard_events: list[str] = ['key_modifier', 'key_press']
    ):
        self.processable_mouse_events = processable_mouse_events
        self.processable_keyboard_events = processable_keyboard_events
    
    def process(self, input_dir: str, output_path: str):
        app_parser = AppDataParser(f"{input_dir}/app_events.json")
        mouse_parser = MouseDataParser(f"{input_dir}/mouse_events.json")
        kb_parser = KeyboardDataParser(f"{input_dir}/keyboard_events.json")

        session_builder = SessionBuilder(
            self.processable_mouse_events,
            self.processable_keyboard_events
        )
        sessions = session_builder.build_sessions(
            app_parser.events,
            mouse_parser.events,
            kb_parser.events
        )

        event_processor = EventProcessor()
        metadata_events = [event_processor.process_session(s) for s in sessions]
        IOUtils.write_metadata(metadata_events, output_path)
        # cleaned = EventDeduplicator.deduplicate(metadata_events)        
        # IOUtils.write_metadata(cleaned, output_path)