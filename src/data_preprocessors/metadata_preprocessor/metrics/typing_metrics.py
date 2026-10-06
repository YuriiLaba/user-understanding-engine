import statistics
from typing import Optional

class TypingMetricsCalculator:
    """Calculates typing speed and consistency metrics."""
    
    @staticmethod
    def calculate(kb_events: list[dict]) -> Optional[dict]:
        """Calculate WPM, KPS, flight time, and consistency."""
        typing_events = [e for e in kb_events if e.get('event_type') == 'key_press']
        
        if len(typing_events) < 2:
            return {
                "wpm": 0.0,
                "kps": 0.0,
                "avg_flight_time": 0.0,
                "flight_time_sd": 0.0,
                "note": "Insufficient data"
            }
        
        # Calculate flight times
        flight_times = []
        for i in range(len(typing_events) - 1):
            try:
                current = typing_events[i]['data']['timestamp_precise']
                next_time = typing_events[i + 1]['data']['timestamp_precise']
                flight_times.append(next_time - current)
            except KeyError:
                continue
        
        # Filter out thinking pauses (> 2s)
        valid_flights = [ft for ft in flight_times if ft < 2.0]
        
        if not valid_flights:
            return {"wpm": 0.0, "kps": 0.0, "avg_flight_time": 0.0, "flight_time_sd": 0.0}
        
        total_duration = sum(valid_flights)
        total_intervals = len(valid_flights)
        avg_flight_time = total_duration / total_intervals
        
        flight_time_sd = statistics.stdev(valid_flights) if total_intervals > 1 else 0.0
        
        total_keystrokes = total_intervals + 1
        kps = total_keystrokes / total_duration if total_duration > 0 else 0
        
        minutes = total_duration / 60
        estimated_words = total_keystrokes / 5
        wpm = estimated_words / minutes if minutes > 0 else 0
        
        return {
            "kps": round(kps, 2),
            "wpm": round(wpm, 2),
            "avg_flight_time": round(avg_flight_time, 4),
            "flight_time_sd": round(flight_time_sd, 4),
            "sample_size": total_intervals
        }