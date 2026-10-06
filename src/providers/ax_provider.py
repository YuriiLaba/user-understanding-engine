import os, json
from tqdm import tqdm
from src.helpers.logger import setup_logging

from src.models.base_model import BaseVisionModel
from src.timestamp_enrichment.context import TimestampContext

class AXProvider:
    """
    AXProvider runs a vision model over cleaned AX records:
    it loads AX, forms sequential pairs, generates model transcriptions
    and short summaries (with limited history), and writes them to .jsonl files.
    """

    def __init__(self, 
                 model: BaseVisionModel, 
                 path_to_store_experiment: str,
                 model_context_size: int,
                 history_size: int,
                 log_filename: str = "logs.log",
                 timestamp_ctx: TimestampContext | None = None,
                 chunk_size: int = 3,
                 ):
        """
        Args:
            model: Vision model implementing `transcribe_axs` and `summarize_axs`
            path_to_store_experiment: Directory to store experiment results
            history_size: How many previous AX frames to include in summaries
            log_filename: Filename for logging
            chunk_size: Pairs submitted per batched model call. Only used when the
                model supports batched generation (SGLang). Keep >= data_parallel_size
                so every GPU stays busy.
        """
        self.model = model
        self.path_to_store_experiment = path_to_store_experiment
        self.model_context_size = model_context_size
        self.history_size = history_size
        self.log_filename = log_filename
        self.timestamp_ctx = timestamp_ctx
        self.chunk_size = chunk_size

        os.makedirs(path_to_store_experiment, exist_ok=True)
        log_filename = os.path.join(self.path_to_store_experiment, log_filename)
        self.logger = setup_logging(log_filename, "AxProvider") 

    # TODO: we should use better tokenizer
    @staticmethod
    def get_ax_from_file(ax_path: str) -> list[dict]:
        """Load processed AX JSON list from disk."""
        with open(ax_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def process_ax_pair(self, ax1: dict, ax2: dict, pair_idx: int, ax_history: list[dict]) -> tuple:
        """
        Run the model on a single AX pair and produce:
        - transcription for the pair
        - short summary conditioned on `ax_history`

        Returns:
            (transcription_entry, summary_entry)
        """
        transcription = self.model.transcribe_axs(ax1, ax2)
        transcription_entry = {
            'pair_idx': pair_idx,
            'ax_pair_ids': (ax1.get("id", -1), ax2.get("id", -1)),
            'transcription': transcription
        }

        stats = self.model.get_response_stats()
        stats = stats.to_log_row()
        self.logger.info(f"[Transcription stats] for pair {pair_idx}: {stats}")

        summary = self.model.summarize_axs(ax_history, (ax1, ax2))
        summary_entry = {
            'pair_idx': pair_idx,
            'ax_pair_ids': (ax1.get("id", -1), ax2.get("id", -1)),
            'history_frames': [ax.get("id", -1) for ax in ax_history],
            'summary': summary
        }

        stats = self.model.get_response_stats()
        stats = stats.to_log_row()
        self.logger.info(f"[Summary stats] for pair {pair_idx}: {stats}")

        return transcription_entry, summary_entry
    
    @staticmethod
    def count_tokens(ax: dict) -> int:
        """Estimate token count for an AX record (simple approximation)."""
        return len(str(ax).split(' '))

    def _build_pairs(self, ax_clean: list[dict]) -> list[tuple[int, dict, dict, list[dict]]]:
        """Form sequential pairs, apply the context-size skip, and precompute
        each pair's summary history window.

        The rolling history and the skip decision both depend only on raw AX
        inputs, so the full schedule (including which pairs are dropped) is known
        upfront in a single cheap pass. Skipped pairs are not added to history,
        matching the original sequential behaviour. Returns
        (pair_idx, ax1, ax2, history_window) per surviving pair.
        """
        pairs: list[tuple[int, dict, dict, list[dict]]] = []
        ax_history: list[dict] = []
        for i in range(0, len(ax_clean) - 1, 2):
            pair_idx = i // 2
            ax1 = ax_clean[i]
            ax2 = ax_clean[i + 1]

            # NOTE: For now, temporary solution: Skip pairs that exceed model context size
            pair_length = AXProvider.count_tokens(ax1) + AXProvider.count_tokens(ax2)
            history_length = sum(AXProvider.count_tokens(ax) for ax in ax_history)
            if pair_length + history_length > self.model_context_size:
                self.logger.info(
                    f"Skipping pair {pair_idx} (frames: {ax1.get('id', -1)}, {ax2.get('id', -1)}) "
                    f"due to history size limits"
                )
                continue

            pairs.append((pair_idx, ax1, ax2, list(ax_history)))
            ax_history.extend([ax1, ax2])
            if len(ax_history) > self.history_size:
                ax_history = ax_history[-self.history_size:]
        return pairs

    def process_ax(self, ax_path: str) -> None:
        """
        Main processing loop:
        - Load cleaned AX records
        - Iterate in pairs (0&1, 2&3, ...)
        - Run model transcription + summary
        - Write .jsonl outputs
        - Maintain rolling AX history for summaries
        """
        ax_clean = self.get_ax_from_file(ax_path)
        print("Loaded AX length:", len(ax_clean))

        pairs = self._build_pairs(ax_clean)

        with open(f"{self.path_to_store_experiment}/transcriptions.jsonl", "w", encoding="utf-8") as tf, \
             open(f"{self.path_to_store_experiment}/summaries.jsonl", "w", encoding="utf-8") as sf:
            # Batched path keeps all data-parallel GPUs busy; models without
            # batch support (MLX/OpenAI) fall back to one-pair-at-a-time.
            if hasattr(self.model, "generate_response_batch"):
                count = self._process_batched(pairs, tf, sf)
            else:
                count = self._process_sequential(pairs, tf, sf)

        print(f"\nCompleted! Processed {count} AX pairs")
        print(f"Results saved to '{self.path_to_store_experiment}'")

    def _enrich_and_write(self, transcription_entry: dict, summary_entry: dict, tf, sf) -> None:
        if self.timestamp_ctx is not None:
            transcription_entry = self.timestamp_ctx.attach_to_entry(transcription_entry)
            summary_entry = self.timestamp_ctx.attach_to_entry(summary_entry)
        tf.write(json.dumps(transcription_entry, ensure_ascii=False) + "\n")
        sf.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _pair_ids(ax1: dict, ax2: dict) -> tuple:
        return (ax1.get("id", -1), ax2.get("id", -1))

    def _process_sequential(self, pairs, tf, sf) -> int:
        processed = 0
        for pair_idx, ax1, ax2, history in tqdm(pairs, total=len(pairs), desc="Processing AX pairs"):
            try:
                transcription_entry, summary_entry = self.process_ax_pair(ax1, ax2, pair_idx, history)
                self._enrich_and_write(transcription_entry, summary_entry, tf, sf)
                processed += 1
            except Exception as e:
                print(f"\nError processing pair {pair_idx}: {e}")
                self.logger.error(
                    f"Error processing pair {pair_idx} (frames: {ax1.get('id', -1)}, {ax2.get('id', -1)}): {str(e)}"
                )
        return processed

    def _process_batched(self, pairs, tf, sf) -> int:
        processed = 0
        total = len(pairs)
        for start in tqdm(range(0, total, self.chunk_size), desc="Processing AX chunks"):
            chunk = pairs[start:start + self.chunk_size]
            try:
                transcriptions = self.model.transcribe_axs_batch(
                    [(ax1, ax2) for _, ax1, ax2, _ in chunk]
                )
                summaries = self.model.summarize_axs_batch(
                    [(history, (ax1, ax2)) for _, ax1, ax2, history in chunk]
                )
            except Exception as e:
                pair_ids = [p[0] for p in chunk]
                self.logger.error(f"Error processing chunk (pairs {pair_ids}): {str(e)}")
                continue

            for (pair_idx, ax1, ax2, history), (t_text, t_stats), (s_text, s_stats) in zip(
                chunk, transcriptions, summaries
            ):
                self.logger.info(f"[Transcription stats] for pair {pair_idx}: {t_stats.to_log_row()}")
                self.logger.info(f"[Summary stats] for pair {pair_idx}: {s_stats.to_log_row()}")

                transcription_entry = {
                    'pair_idx': pair_idx,
                    'ax_pair_ids': self._pair_ids(ax1, ax2),
                    'transcription': t_text,
                }
                summary_entry = {
                    'pair_idx': pair_idx,
                    'ax_pair_ids': self._pair_ids(ax1, ax2),
                    'history_frames': [ax.get("id", -1) for ax in history],
                    'summary': s_text,
                }
                self._enrich_and_write(transcription_entry, summary_entry, tf, sf)
                processed += 1
        return processed
