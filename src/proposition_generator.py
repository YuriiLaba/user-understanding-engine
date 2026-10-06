import json
from typing import List
from src.helpers.logger import setup_logging
from src.prompts import PROPOSE_PROMPT

from src.models.base_model import BaseVisionModel
from src.data_models import Observation
from src.timestamp_enrichment.context import batch_timestamp_range


class PropositionGenerator:
    def __init__(
        self,
        model: BaseVisionModel,
        user_name: str = "the user",
        k_items_per_batch: int = 5,
        chunk_size: int = 3,
        log_filename: str = "logs.log"
    ):
        """Initialize the proposition generator.

        Args:
            model: Vision model instance
            user_name: Name of the user for personalization
            k_items_per_batch: Number of transcription+summary pairs to batch together
            chunk_size: Number of observations submitted per batched model call.
                Only used when the model supports batched generation (e.g. SGLang
                with data_parallel_size > 1). Should be >= data_parallel_size so
                every replica/GPU stays busy; a few times larger helps pipelining.
            log_filename: Filename for logging
        """
        self.model = model
        self.user_name = user_name
        self.k_items_per_batch = k_items_per_batch
        self.chunk_size = chunk_size
        self.propose_prompt = PROPOSE_PROMPT
        self.logger = setup_logging(log_filename, "PropositionProvider") 
        

    def load_data(
        self,
        summaries_path: str,
        transcriptions_path: str,
    ) -> tuple[dict, dict]:
        """Load summaries and transcriptions from files.
        
        Args:
            summaries_path: Path to summaries JSON file
            transcriptions_path: Path to transcriptions JSONL file
            
        Returns:
            Tuple of (summaries_dict, transcriptions_dict) indexed by pair_idx
        """
        summaries = {}
        with open(summaries_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    summaries[item['pair_idx']] = item
                except json.JSONDecodeError as e:
                    continue
        
        transcriptions = {}
        with open(transcriptions_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    transcriptions[item['pair_idx']] = item
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: {line}. Error: {e}")
                    continue
        
        return summaries, transcriptions

    def create_observation_batches(
        self,
        summaries: dict,
        transcriptions: dict
    ) -> List[Observation]:
        """Create batches of k transcriptions + summaries combined.
        
        Args:
            summaries: Dictionary of summaries indexed by pair_idx
            transcriptions: Dictionary of transcriptions indexed by pair_idx
            
        Returns:
            List of Update objects, each containing k combined items
        """

        common_indices = sorted(set(summaries.keys()) & set(transcriptions.keys()))
        
        observations = []
        for i in range(0, len(common_indices), self.k_items_per_batch):
            batch_indices = common_indices[i:i + self.k_items_per_batch]
            
            batch_content = []
            for idx in batch_indices:
                summary = summaries[idx].get('summary', '')
                transcription = transcriptions[idx].get('transcription', '')
                
                combined = f"--- Pair {idx} ---\n\n"
                if summary:
                    combined += f"## Summary\n{summary}\n\n"
                if transcription:
                    combined += f"## Transcription\n{transcription}\n\n"
                
                batch_content.append(combined)
            
            observation = Observation(
                content="\n".join(batch_content),
                pair_indices=batch_indices
            )
            observations.append(observation)
        
        return observations

    def process(
        self,
        summaries_path: str,
        transcriptions_path: str,
        output_path: str,
        *,
        enrich_timestamps: bool = False,
        modality: str | None = None,
    ):
        """Process all updates and save propositions to file.
        
        Args:
            summaries_path: Path to summaries JSON file
            transcriptions_path: Path to transcriptions JSONL file
            output_path: Path to save generated propositions
        """
        summaries, transcriptions = self.load_data(summaries_path, transcriptions_path)

        updates = self.create_observation_batches(summaries, transcriptions)

        # A model that exposes batched generation (SGLang with data_parallel_size
        # > 1) only keeps all its GPUs busy when several requests are in flight at
        # once. Submitting one at a time leaves every replica but one idle, so we
        # send chunks of `chunk_size` per call. Models without batch support fall
        # back to the original one-at-a-time loop.
        if hasattr(self.model, "generate_propositions_batch"):
            self._process_batched(
                updates, summaries, output_path,
                enrich_timestamps=enrich_timestamps, modality=modality,
            )
        else:
            self._process_sequential(
                updates, summaries, output_path,
                enrich_timestamps=enrich_timestamps, modality=modality,
            )

    def _process_sequential(self, updates, summaries, output_path, *, enrich_timestamps, modality):
        for i, update in enumerate(updates):
            print(f"Processing batch {i+1}/{len(updates)} "
                  f"(pairs: {update.pair_indices})...")

            try:
                propositions = self.model.generate_propositions(update, self.propose_prompt, self.user_name)

                stats = self.model.get_response_stats()
                stats = stats.to_log_row()
                self.logger.info(f"[Providers stats] for pair {update.pair_indices}: {stats}")

                result = {
                    "batch_id": i,
                    "pair_indices": update.pair_indices,
                    "propositions": propositions["propositions"]
                }

            except Exception as e:
                self.logger.error(
                    f"Batch {i} (pairs {update.pair_indices}) failed: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                print(f"  Error processing batch {i}: {type(e).__name__}: {e}")
                result = {
                    "batch_id": i,
                    "pair_indices": update.pair_indices,
                    "error": f"{type(e).__name__}: {e}"[:500],
                }

            self._finalize_and_write(
                result, update, summaries, output_path,
                enrich_timestamps=enrich_timestamps, modality=modality,
            )

    def _process_batched(self, updates, summaries, output_path, *, enrich_timestamps, modality):
        total = len(updates)
        for start in range(0, total, self.chunk_size):
            chunk = updates[start:start + self.chunk_size]
            print(f"Processing chunk {start // self.chunk_size + 1} "
                  f"(batches {start + 1}-{start + len(chunk)}/{total}) across GPUs...")

            try:
                batch_results = self.model.generate_propositions_batch(
                    chunk, self.propose_prompt, self.user_name
                )
            except Exception as e:
                # A whole-chunk failure (e.g. engine error) shouldn't abort the
                # run — record the error for every item in the chunk and continue.
                self.logger.error(
                    f"Chunk starting at batch {start} failed: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                print(f"  Error processing chunk starting at batch {start}: {type(e).__name__}: {e}")
                batch_results = [{"error": f"{type(e).__name__}: {e}"[:500]} for _ in chunk]

            for offset, (update, item) in enumerate(zip(chunk, batch_results)):
                batch_id = start + offset
                if "error" in item:
                    self.logger.error(
                        f"Batch {batch_id} (pairs {update.pair_indices}) failed: {item['error']}"
                    )
                    result = {
                        "batch_id": batch_id,
                        "pair_indices": update.pair_indices,
                        "error": item["error"],
                    }
                else:
                    stats = item.get("stats")
                    if stats is not None:
                        self.logger.info(
                            f"[Providers stats] for pair {update.pair_indices}: {stats.to_log_row()}"
                        )
                    result = {
                        "batch_id": batch_id,
                        "pair_indices": update.pair_indices,
                        "propositions": item["propositions"],
                    }

                self._finalize_and_write(
                    result, update, summaries, output_path,
                    enrich_timestamps=enrich_timestamps, modality=modality,
                )

    def _finalize_and_write(self, result, update, summaries, output_path, *, enrich_timestamps, modality):
        if enrich_timestamps and modality is not None:
            ts_range = batch_timestamp_range(update.pair_indices, summaries)
            if ts_range is not None:
                result["timestamp_range"] = ts_range
            result["modality"] = modality

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')