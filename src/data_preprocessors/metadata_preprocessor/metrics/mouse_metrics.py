import math
from typing import Optional
from src.data_preprocessors.metadata_preprocessor.utils.time_utils import TimeUtils

class MouseMetricsCalculator:
    """Calculates mouse usage metrics."""
    
    @staticmethod
    def calculate(mouse_events: list[dict]) -> Optional[dict]:
        """Calculate distance, speed, clicks, and screen area usage."""
        if not mouse_events or len(mouse_events) < 2:
            return None
        
        events = sorted(mouse_events, key=lambda x: x['timestamp'])
        
        total_distance = 0.0
        click_count = 0
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        last_x, last_y = None, None
        
        start_time = TimeUtils.parse_iso_to_timestamp(events[0]['timestamp'])
        
        for e in events:
            data = e['data']
            x, y = data.get('x'), data.get('y')
            
            # Track bounding box
            if x is not None and y is not None:
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
            
            # Count clicks
            if e['event_type'] == 'mouse_click' and data.get('action') == 'down':
                click_count += 1
            
            # Calculate movement distance
            elif e['event_type'] == 'mouse_move' and x is not None and y is not None:
                if last_x is not None:
                    dist = math.hypot(x - last_x, y - last_y)
                    if dist < 3000:  # Filter OS glitches
                        total_distance += dist
                last_x, last_y = x, y
        
        end_time = TimeUtils.parse_iso_to_timestamp(events[-1]['timestamp'])
        duration = end_time - start_time
        avg_speed = total_distance / duration if duration > 0 else 0
        
        # Calculate screen area
        if min_x != float('inf'):
            area_used = (max_x - min_x) * (max_y - min_y)
        else:
            area_used = 0.0
        
        return {
            "total_distance_px": round(total_distance, 2),
            "avg_speed_px_s": round(avg_speed, 2),
            "click_count": click_count,
            "screen_area_used": round(area_used, 2)
        }