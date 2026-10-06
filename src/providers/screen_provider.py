import os
import json
from typing import List
from tqdm import tqdm
from src.models.base_model import BaseVisionModel
from src.helpers.logger import setup_logging
from src.timestamp_enrichment.context import TimestampContext


class ScreenProvider:    
    def __init__(self, model: BaseVisionModel, 
                 path_to_store_experiment: str,
                 history_size: int,
                 apply_ocr: bool = False,
                 log_filename: str = "logs.log",
                 timestamp_ctx: TimestampContext | None = None,
                 chunk_size: int = 3,
                 ):
        """
        Initialize frame processor

        Args:
            model: Vision model instance
            path_to_store_experiment: Directory to store experiment results
            history_size: Number of historical frames to keep
            apply_ocr: Whether to apply OCR before processing
            log_filename: Filename for logging
            chunk_size: Frame pairs submitted per batched model call. Only used
                when the model supports batched generation (SGLang). Keep
                >= data_parallel_size so every GPU stays busy.
        """
        self.model = model
        self.path_to_store_experiment = path_to_store_experiment
        self.history_size = history_size
        self.apply_ocr = apply_ocr
        self.log_filename = log_filename
        self.timestamp_ctx = timestamp_ctx
        self.chunk_size = chunk_size

        os.makedirs(path_to_store_experiment, exist_ok=True)
        log_filename = os.path.join(self.path_to_store_experiment, log_filename)
        self.logger = setup_logging(log_filename, "ScreenProvider") 
        self.SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.gif', '.webp')

    
    def get_frame_files(self, frames_dir: str) -> List[str]:
        """
        Get sorted list of frame files
        
        Args:
            frames_dir: Directory containing frames
            
        Returns:
            Sorted list of frame filenames
        """
        return sorted([
            f for f in os.listdir(frames_dir) 
            if f.endswith(self.SUPPORTED_IMAGE_FORMATS)
        ])
    
    def process_frame_pair(self, frame1_path: str, frame2_path: str,
                          pair_idx: int, frame_history: List[str]) -> tuple:
        """
        Process a single frame pair
        
        Args:
            frame1_path: Path to first frame
            frame2_path: Path to second frame
            pair_idx: Index of the frame pair
            frame_history: List of historical frame paths
            
        Returns:
            Tuple of (transcription_entry, summary_entry)
        """
        # Transcribe
        transcription = self.model.transcribe_frames(
            frame1_path, frame2_path, self.apply_ocr
        )
        
        transcription_entry = {
            'pair_idx': pair_idx,
            'frames': (os.path.basename(frame1_path), os.path.basename(frame2_path)),
            'transcription': transcription
        }

        stats = self.model.get_response_stats()
        stats = stats.to_log_row()
        self.logger.info(f"[Transcription stats] for pair {pair_idx}: {stats}")
        
        # Summarize
        summary = self.model.summarize_frames(
            frame_history, (frame1_path, frame2_path), self.apply_ocr
        )
        
        summary_entry = {
            'pair_idx': pair_idx,
            'frames': (os.path.basename(frame1_path), os.path.basename(frame2_path)),
            'history_frames': [os.path.basename(f) for f in frame_history],
            'summary': summary
        }

        stats = self.model.get_response_stats()
        stats = stats.to_log_row()
        self.logger.info(f"[Summary stats] for pair {pair_idx}: {stats}")
        
        return transcription_entry, summary_entry
    
    def _build_pairs(self, frame_files: List[str], frames_dir: str) -> list[tuple[int, str, str, list[str]]]:
        """Form sequential frame pairs and precompute each summary's history window.

        The rolling history is built from raw frame paths, so the schedule is
        known upfront. Each window is a snapshot taken BEFORE the current pair is
        added, matching the original sequential behaviour. Returns
        (pair_idx, frame1, frame2, history_window) per pair.
        """
        pairs: list[tuple[int, str, str, list[str]]] = []
        frame_history: list[str] = []
        for i in range(0, len(frame_files) - 1, 2):
            pair_idx = i // 2
            frame1 = os.path.join(frames_dir, frame_files[i])
            frame2 = os.path.join(frames_dir, frame_files[i + 1])
            pairs.append((pair_idx, frame1, frame2, list(frame_history)))

            frame_history.extend([frame1, frame2])
            if len(frame_history) > self.history_size:
                frame_history = frame_history[-self.history_size:]
        return pairs

    def process_frames(self, frames_dir: str) -> None:
        """
        Process all frames in directory

        Args:
            frames_dir: Directory containing frame images

        Returns:
            Dictionary with processing statistics
        """
        frame_files = self.get_frame_files(frames_dir)

        if len(frame_files) < 2:
            print("Not enough frames to process (need at least 2)")
            return

        pairs = self._build_pairs(frame_files, frames_dir)

        with open(f"{self.path_to_store_experiment}/transcriptions.jsonl", "w", encoding="utf-8") as tf, \
             open(f"{self.path_to_store_experiment}/summaries.jsonl", "w", encoding="utf-8") as sf:
            # Batched path keeps all data-parallel GPUs busy; models without
            # batch support (MLX/OpenAI) fall back to one-pair-at-a-time.
            if hasattr(self.model, "generate_response_batch"):
                count = self._process_batched(pairs, tf, sf)
            else:
                count = self._process_sequential(pairs, tf, sf)

        print(f"\nCompleted! Processed {count} frame pairs")
        print(f"Results saved to '{self.path_to_store_experiment}'")

    def _enrich_and_write(self, transcription_entry: dict, summary_entry: dict, tf, sf) -> None:
        if self.timestamp_ctx is not None:
            transcription_entry = self.timestamp_ctx.attach_to_entry(transcription_entry)
            summary_entry = self.timestamp_ctx.attach_to_entry(summary_entry)
        tf.write(json.dumps(transcription_entry, ensure_ascii=False) + "\n")
        sf.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")

    def _process_sequential(self, pairs, tf, sf) -> int:
        processed = 0
        for pair_idx, frame1, frame2, history in tqdm(pairs, total=len(pairs), desc="Processing frame pairs"):
            try:
                transcription_entry, summary_entry = self.process_frame_pair(
                    frame1, frame2, pair_idx, history
                )
                self._enrich_and_write(transcription_entry, summary_entry, tf, sf)
                processed += 1
            except Exception as e:
                self.logger.error(
                    f"Error processing pair {pair_idx} "
                    f"(frames: {os.path.basename(frame1)}, {os.path.basename(frame2)}): {str(e)}"
                )
        return processed

    def _process_batched(self, pairs, tf, sf) -> int:
        processed = 0
        total = len(pairs)
        for start in tqdm(range(0, total, self.chunk_size), desc="Processing frame chunks"):
            chunk = pairs[start:start + self.chunk_size]
            try:
                transcriptions = self.model.transcribe_frames_batch(
                    [(f1, f2) for _, f1, f2, _ in chunk], self.apply_ocr
                )
                summaries = self.model.summarize_frames_batch(
                    [(history, (f1, f2)) for _, f1, f2, history in chunk], self.apply_ocr
                )
            except Exception as e:
                pair_ids = [p[0] for p in chunk]
                self.logger.error(f"Error processing chunk (pairs {pair_ids}): {str(e)}")
                continue

            for (pair_idx, frame1, frame2, history), (t_text, t_stats), (s_text, s_stats) in zip(
                chunk, transcriptions, summaries
            ):
                self.logger.info(f"[Transcription stats] for pair {pair_idx}: {t_stats.to_log_row()}")
                self.logger.info(f"[Summary stats] for pair {pair_idx}: {s_stats.to_log_row()}")

                transcription_entry = {
                    'pair_idx': pair_idx,
                    'frames': (os.path.basename(frame1), os.path.basename(frame2)),
                    'transcription': t_text,
                }
                summary_entry = {
                    'pair_idx': pair_idx,
                    'frames': (os.path.basename(frame1), os.path.basename(frame2)),
                    'history_frames': [os.path.basename(f) for f in history],
                    'summary': s_text,
                }
                self._enrich_and_write(transcription_entry, summary_entry, tf, sf)
                processed += 1
        return processed