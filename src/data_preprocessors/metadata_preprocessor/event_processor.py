from src.helpers.text_merger import choose_real_text # TODO: this is hotfix module, should be removed
from src.data_models import Session, MetadataEvent
from src.data_preprocessors.metadata_preprocessor.parsers.app_parser import AppDataParser
from src.data_preprocessors.metadata_preprocessor.metrics.mouse_metrics import MouseMetricsCalculator
from src.data_preprocessors.metadata_preprocessor.metrics.typing_metrics import TypingMetricsCalculator
from src.data_preprocessors.metadata_preprocessor.utils.time_utils import TimeUtils
from src.data_preprocessors.metadata_preprocessor.utils.keyboard_text_reconstruction import KeyboardTextReconstructor
from src.data_preprocessors.metadata_preprocessor.utils.layout_decoder import LayoutDecoder

class EventProcessor:
    """Processes sessions into MetadataEvents."""
    
    def __init__(self):
        self.mouse_calc = MouseMetricsCalculator()
        self.typing_calc = TypingMetricsCalculator()
        self.text_reconstructor = KeyboardTextReconstructor()
        self.layout_decoder = LayoutDecoder() # TODO: this is hotfix module, should be removed
    
    def process_session(self, session: Session) -> MetadataEvent:
        """Convert a Session into a MetadataEvent."""
        app_data = session.app_event['data']
        
        # Extract app info
        app_name = app_data['app_name']
        trigger_action = app_data['action']
        windows_names = AppDataParser.extract_window_titles(session.app_event)
        
        # Calculate duration
        duration = TimeUtils.calculate_duration(session.start, session.end)
        
        # Calculate mouse metrics
        mouse_metrics = None
        if session.mouse_events:
            mouse_metrics = self.mouse_calc.calculate(session.mouse_events)
        
        # Calculate keyboard metrics
        typing_metrics = None
        keyboard_transcription = None
        
        if session.kb_events:
            typing_metrics = self.typing_calc.calculate(session.kb_events)
            
            # Transcribe keyboard
            original = self.text_reconstructor.reconstruct(session.kb_events)
            decoded = self.layout_decoder.decode(original)
            real_text = choose_real_text(original, decoded)
            
            keyboard_transcription = {
                "main_text": real_text,
                # "original": original,
                # "decoded": decoded
            }
        
        return MetadataEvent(
            id=session.id,
            duration=duration,
            trigger_action=trigger_action,
            app_name=app_name,
            windows_names=windows_names,
            mouse_metrics=mouse_metrics,
            typing_metrics=typing_metrics,
            keyboard_transcription=keyboard_transcription
        )