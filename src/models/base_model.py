from abc import abstractmethod
from typing import List
from src.data_models import Observation, PropositionItem
from src.models.utils.ocr_processors import OCRProcessor
from src.models.utils.prompt_builder import PromptBuilder
from src.data_models import GenerationStats

from src.prompts import (
    TRANSCRIPTION_PROMPT_OCR,
    SUMMARY_PROMPT_OCR,
    TRANSCRIPTION_PROMPT,
    SUMMARY_PROMPT,
    TRANSCRIPTION_PROMPT_AX,
    SUMMARY_PROMPT_AX,
    TRANSCRIPTION_PROMPT_METADATA,
    SUMMARY_PROMPT_METADATA,
)

class BaseVisionModel:
    def __init__(self, model_identifier: str):
        """Initialize the vision model.
        
        Args:
            model_identifier: Model path or identifier
        """
        self.model_identifier = model_identifier
        self.ocr_processor = OCRProcessor()
        self.prompt_builder = PromptBuilder()
        self._load_model()
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model (implementation specific)."""
        pass
    
    @abstractmethod
    def generate_response(
        self, prompt: str, **kwargs
    ) -> str:
        """Generate response from images and prompt.
        
        Args:
            prompt: Text prompt
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_propositions(
        self, observation: Observation, propose_prompt: str, user_name: str) -> List[PropositionItem]:
        """Generate propositions from an observation.
        
        Args:
            observation: The observation to generate propositions from
            propose_prompt: The prompt template for proposing
            user_name: Name of the user for personalization
            
        Returns:
            List of generated propositions
        """
        pass

    @abstractmethod
    def get_response_stats(self) -> GenerationStats:
        """Get statistics about the last generation.
        
        Returns:
            GenerationStats of generation statistics
        """
        pass
    
    def transcribe_frames(
        self, frame1: str, frame2: str, apply_ocr: bool = False
    ) -> str:
        """Transcribe a pair of frames.
        
        Args:
            frame1: Path to first frame
            frame2: Path to second frame
            apply_ocr: Whether to apply OCR preprocessing
            
        Returns:
            Transcription text
        """
        if apply_ocr:
            texts = [
                self.ocr_processor.extract_text(frame1),
                self.ocr_processor.extract_text(frame2),
            ]
            prompt = self.prompt_builder.build_ocr_prompt(
                texts, TRANSCRIPTION_PROMPT_OCR
            )
            return self.generate_response(prompt=prompt)
        
        return self.generate_response(
            prompt=TRANSCRIPTION_PROMPT, images=[frame1, frame2]
        )
    
    def summarize_frames(
        self,
        history_frames: List[str],
        current_pair: tuple[str, str],
        apply_ocr: bool = False,
    ) -> str:
        """Summarize frames with history.
        
        Args:
            history_frames: List of historical frame paths
            current_pair: Tuple of (frame1, frame2)
            apply_ocr: Whether to apply OCR preprocessing
            
        Returns:
            Summary text
        """
        if apply_ocr:
            all_frame_paths = history_frames + list(current_pair)
            texts = [
                self.ocr_processor.extract_text(frame)
                for frame in all_frame_paths
            ]
            prompt = self.prompt_builder.build_ocr_prompt(
                texts, SUMMARY_PROMPT_OCR
            )
            return self.generate_response(prompt=prompt)
        
        all_images = history_frames + list(current_pair)
        generated_response = self.generate_response(prompt=SUMMARY_PROMPT, images=all_images)
        return generated_response
    
    def transcribe_axs(self, ax1: dict, ax2: dict) -> str:
        """Transcribe a pair of accessibility records.
        
        Args:
            ax1: First accessibility dictionary
            ax2: Second accessibility dictionary
            
        Returns:
            Transcription text
        """
        prompt = self.prompt_builder.build_ax_prompt(
            [ax1, ax2], TRANSCRIPTION_PROMPT_AX
        )
        return self.generate_response(prompt=prompt)
    
    def summarize_axs(
        self, history_axs: List[dict], current_pair: tuple[dict, dict]
    ) -> str:
        """Summarize accessibility records with history.
        
        Args:
            history_axs: List of historical accessibility data
            current_pair: Tuple of (ax1, ax2)
            
        Returns:
            Summary text
        """
        all_axs = history_axs + list(current_pair)
        prompt = self.prompt_builder.build_ax_prompt(all_axs, SUMMARY_PROMPT_AX)
        return self.generate_response(prompt=prompt)
    
    def transcribe_frames_batch(
        self, pairs: List[tuple[str, str]], apply_ocr: bool = False
    ) -> list[tuple[str, GenerationStats]]:
        """Transcribe many frame pairs in one batched model call.

        In OCR mode the batch is text-only (OCR text embedded in the prompt);
        otherwise each prompt carries its two frames as images. Requires a model
        implementing ``generate_response_batch``. Returns (text, stats) per pair.
        """
        if apply_ocr:
            prompts = [
                self.prompt_builder.build_ocr_prompt(
                    [self.ocr_processor.extract_text(f1), self.ocr_processor.extract_text(f2)],
                    TRANSCRIPTION_PROMPT_OCR,
                )
                for f1, f2 in pairs
            ]
            return self.generate_response_batch(prompts)

        prompts = [TRANSCRIPTION_PROMPT for _ in pairs]
        images_list = [[f1, f2] for f1, f2 in pairs]
        return self.generate_response_batch(prompts, images_list=images_list)

    def summarize_frames_batch(
        self, items: List[tuple[List[str], tuple[str, str]]], apply_ocr: bool = False
    ) -> list[tuple[str, GenerationStats]]:
        """Summarize many frame pairs in one batched model call.

        Each item is (history_frames, (frame1, frame2)); history windows are
        precomputed by the caller. OCR mode is text-only; otherwise history +
        current frames are passed as images. Requires ``generate_response_batch``.
        Returns (summary_text, stats) per item, in order.
        """
        if apply_ocr:
            prompts = [
                self.prompt_builder.build_ocr_prompt(
                    [self.ocr_processor.extract_text(f) for f in list(history) + [f1, f2]],
                    SUMMARY_PROMPT_OCR,
                )
                for history, (f1, f2) in items
            ]
            return self.generate_response_batch(prompts)

        prompts = [SUMMARY_PROMPT for _ in items]
        images_list = [list(history) + [f1, f2] for history, (f1, f2) in items]
        return self.generate_response_batch(prompts, images_list=images_list)

    def transcribe_axs_batch(self, pairs: List[tuple[dict, dict]]) -> list[tuple[str, GenerationStats]]:
        """Transcribe many AX pairs in one batched (text-only) model call.

        Requires a model implementing ``generate_response_batch``. Returns
        (transcription_text, stats) per pair, in order.
        """
        prompts = [
            self.prompt_builder.build_ax_prompt([ax1, ax2], TRANSCRIPTION_PROMPT_AX)
            for ax1, ax2 in pairs
        ]
        return self.generate_response_batch(prompts)

    def summarize_axs_batch(
        self, items: List[tuple[List[dict], tuple[dict, dict]]]
    ) -> list[tuple[str, GenerationStats]]:
        """Summarize many AX pairs in one batched (text-only) model call.

        Each item is (history_records, (ax1, ax2)); history windows are
        precomputed by the caller. Requires ``generate_response_batch``. Returns
        (summary_text, stats) per item, in order.
        """
        prompts = [
            self.prompt_builder.build_ax_prompt(list(history) + [ax1, ax2], SUMMARY_PROMPT_AX)
            for history, (ax1, ax2) in items
        ]
        return self.generate_response_batch(prompts)

    def transcribe_metadatas_batch(self, pairs: List[tuple[dict, dict]]) -> list[tuple[str, GenerationStats]]:
        """Transcribe many metadata pairs in one batched model call.

        Each pair is independent, so this maps directly onto the batched
        backend. Requires a model implementing ``generate_response_batch``.
        Returns (transcription_text, stats) per pair, in order.
        """
        prompts = [
            self.prompt_builder.build_metadata_prompt([md1, md2], TRANSCRIPTION_PROMPT_METADATA)
            for md1, md2 in pairs
        ]
        return self.generate_response_batch(prompts)

    def summarize_metadatas_batch(
        self, items: List[tuple[List[dict], tuple[dict, dict]]]
    ) -> list[tuple[str, GenerationStats]]:
        """Summarize many metadata pairs in one batched model call.

        Each item is (history_records, (md1, md2)); the history windows are
        precomputed by the caller from raw inputs, so the summaries carry no
        dependency on each other's outputs and can run concurrently. Requires a
        model implementing ``generate_response_batch``. Returns (summary_text,
        stats) per item, in order.
        """
        prompts = [
            self.prompt_builder.build_metadata_prompt(list(history) + [md1, md2], SUMMARY_PROMPT_METADATA)
            for history, (md1, md2) in items
        ]
        return self.generate_response_batch(prompts)

    def transcribe_metadatas(self, metadata1: dict, metadata2: dict) -> str:
        """Transcribe a pair of metadata records.
        
        Args:
            metadata1: First metadata dictionary
            metadata2: Second metadata dictionary
            
        Returns:
            Transcription text
        """
        prompt = self.prompt_builder.build_metadata_prompt(
            [metadata1, metadata2], TRANSCRIPTION_PROMPT_METADATA
        )
        return self.generate_response(prompt=prompt)
    
    def summarize_metadatas(
        self, history_metadatas: List[dict], current_pair: tuple[dict, dict]
    ) -> str:
        """Summarize metadata records with history.
        
        Args:
            history_metadatas: List of historical metadata records
            current_pair: Tuple of (metadata1, metadata2)
            
        Returns:
            Summary text
        """
        all_metadatas = history_metadatas + list(current_pair)
        prompt = self.prompt_builder.build_metadata_prompt(
            all_metadatas, SUMMARY_PROMPT_METADATA
        )
        return self.generate_response(prompt=prompt)