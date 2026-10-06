import os, json
from tqdm import tqdm
from src.helpers.logger import setup_logging

from src.models.base_model import BaseVisionModel
from src.timestamp_enrichment.context import TimestampContext

class MetadataProvider:
    def __init__(self,
                 model: BaseVisionModel,
                 path_to_store_experiment: str,
                 history_size: int,
                 log_filename: str = "logs.log",
                 timestamp_ctx: TimestampContext | None = None,
                 chunk_size: int = 3,
                 ):
        """
        Args:
            model: Vision model implementing `transcribe_axs` and `summarize_axs`
            path_to_store_experiment: Directory to store experiment results
            history_size: How many previous metadata records to include in summaries
            log_filename: Filename for logging
            chunk_size: Pairs submitted per batched model call. Only used when the
                model supports batched generation (SGLang). Keep >= data_parallel_size
                so every GPU stays busy.
        """
        self.model = model
        self.path_to_store_experiment = path_to_store_experiment
        self.history_size = history_size
        self.log_filename = log_filename
        self.timestamp_ctx = timestamp_ctx
        self.chunk_size = chunk_size

        os.makedirs(path_to_store_experiment, exist_ok=True)
        log_filename = os.path.join(self.path_to_store_experiment, log_filename)
        self.logger = setup_logging(log_filename, "MetadataProvider")


    @staticmethod
    def get_metadata_from_file(metadata_path: str) -> list[dict]:
        """Load processed Metadata JSON list from disk."""
        with open(metadata_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def process_metadata_pair(self, metadata1: dict, metadata2: dict, pair_idx: int, meatadata_history: list[dict]) -> tuple:
        """
        Run the model on a single metadata pair and produce:
        - transcription for the pair
        - short summary conditioned on `meatadata_history`

        Returns:
            (transcription_entry, summary_entry)
        """
        transcription = self.model.transcribe_metadatas(metadata1, metadata2)
        transcription_entry = {
            'pair_idx': pair_idx,
            'metadata_pair_ids': (metadata1.get("id", -1), metadata2.get("id", -1)),
            'transcription': transcription
        }

        stats = self.model.get_response_stats()
        stats = stats.to_log_row()
        self.logger.info(f"[Transcription stats] for pair {pair_idx}: {stats}")

        summary = self.model.summarize_metadatas(meatadata_history, (metadata1, metadata2))
        summary_entry = {
            'pair_idx': pair_idx,
            'metadata_pair_ids': (metadata1.get("id", -1), metadata2.get("id", -1)),
            'history_frames': [ax.get("id", -1) for ax in meatadata_history],
            'summary': summary
        }

        stats = self.model.get_response_stats()
        stats = stats.to_log_row()
        self.logger.info(f"[Summary stats] for pair {pair_idx}: {stats}")

        return transcription_entry, summary_entry

    def _build_pairs(self, metadata_clean: list[dict]) -> list[tuple[int, dict, dict, list[dict]]]:
        """Form sequential pairs and precompute each pair's summary history window.

        The rolling history is built from raw metadata records (not model
        outputs), so the whole schedule is known upfront in a single cheap pass.
        Each summary window is a snapshot taken BEFORE the current pair is added,
        matching the original sequential behaviour. Returns
        (pair_idx, md_1, md_2, history_window) per pair.
        """
        pairs: list[tuple[int, dict, dict, list[dict]]] = []
        metadata_history: list[dict] = []
        for i in range(0, len(metadata_clean) - 1, 2):
            md_1 = metadata_clean[i]
            md_2 = metadata_clean[i + 1]
            pairs.append((i // 2, md_1, md_2, list(metadata_history)))

            metadata_history.extend([md_1, md_2])
            if len(metadata_history) > self.history_size:
                metadata_history = metadata_history[-self.history_size:]
        return pairs

    def process_metadata(self, metadata_path: str) -> None:
        """
        Main processing loop:
        - Load cleaned metadata records
        - Iterate in pairs (0&1, 2&3, ...)
        - Run model transcription + summary
        - Write .jsonl outputs
        - Maintain rolling metadata history for summaries
        """
        metadata_clean = self.get_metadata_from_file(metadata_path)
        print("Loaded MetaData length:", len(metadata_clean))

        pairs = self._build_pairs(metadata_clean)

        with open(f"{self.path_to_store_experiment}/transcriptions.jsonl", "w", encoding="utf-8") as tf, \
             open(f"{self.path_to_store_experiment}/summaries.jsonl", "w", encoding="utf-8") as sf:
            # Batched path keeps all data-parallel GPUs busy; models without
            # batch support (MLX/OpenAI) fall back to one-pair-at-a-time.
            if hasattr(self.model, "generate_response_batch"):
                count = self._process_batched(pairs, tf, sf)
            else:
                count = self._process_sequential(pairs, tf, sf)

        print(f"\nCompleted! Processed {count} MetaData pairs")
        print(f"Results saved to '{self.path_to_store_experiment}'")

    def _enrich_and_write(self, transcription_entry: dict, summary_entry: dict, tf, sf) -> None:
        if self.timestamp_ctx is not None:
            transcription_entry = self.timestamp_ctx.attach_to_entry(transcription_entry)
            summary_entry = self.timestamp_ctx.attach_to_entry(summary_entry)
        tf.write(json.dumps(transcription_entry, ensure_ascii=False) + "\n")
        sf.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _pair_ids(md_1: dict, md_2: dict) -> tuple:
        return (md_1.get("id", -1), md_2.get("id", -1))

    def _process_sequential(self, pairs, tf, sf) -> int:
        processed = 0
        for pair_idx, md_1, md_2, history in tqdm(pairs, total=len(pairs), desc="Processing MetaData pairs"):
            try:
                transcription_entry, summary_entry = self.process_metadata_pair(
                    md_1, md_2, pair_idx, history
                )
                self._enrich_and_write(transcription_entry, summary_entry, tf, sf)
                processed += 1
            except Exception as e:
                error_msg = f"Error processing pair {pair_idx} (frames: {md_1}, {md_2}): {str(e)}"
                self.logger.error(error_msg)
        return processed

    def _process_batched(self, pairs, tf, sf) -> int:
        processed = 0
        total = len(pairs)
        for start in tqdm(range(0, total, self.chunk_size), desc="Processing MetaData chunks"):
            chunk = pairs[start:start + self.chunk_size]
            try:
                transcriptions = self.model.transcribe_metadatas_batch(
                    [(md_1, md_2) for _, md_1, md_2, _ in chunk]
                )
                summaries = self.model.summarize_metadatas_batch(
                    [(history, (md_1, md_2)) for _, md_1, md_2, history in chunk]
                )
            except Exception as e:
                pair_ids = [p[0] for p in chunk]
                self.logger.error(f"Error processing chunk (pairs {pair_ids}): {str(e)}")
                continue

            for (pair_idx, md_1, md_2, history), (t_text, t_stats), (s_text, s_stats) in zip(
                chunk, transcriptions, summaries
            ):
                self.logger.info(f"[Transcription stats] for pair {pair_idx}: {t_stats.to_log_row()}")
                self.logger.info(f"[Summary stats] for pair {pair_idx}: {s_stats.to_log_row()}")

                transcription_entry = {
                    'pair_idx': pair_idx,
                    'metadata_pair_ids': self._pair_ids(md_1, md_2),
                    'transcription': t_text,
                }
                summary_entry = {
                    'pair_idx': pair_idx,
                    'metadata_pair_ids': self._pair_ids(md_1, md_2),
                    'history_frames': [md.get("id", -1) for md in history],
                    'summary': s_text,
                }
                self._enrich_and_write(transcription_entry, summary_entry, tf, sf)
                processed += 1
        return processed
