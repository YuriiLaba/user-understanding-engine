from typing import List, Optional

class PromptBuilder:
    @staticmethod
    def build_ocr_prompt(texts: List[str], base_prompt: str) -> str:
        """Build a prompt with OCR text blocks.
        
        Args:
            texts: List of OCR-extracted texts
            base_prompt: Base prompt template
            
        Returns:
            Formatted prompt string
        """
        blocks = [
            f"[SCREENSHOT {idx}]\n{'-' * 20}\n{text}"
            for idx, text in enumerate(texts, start=1)
        ]
        
        return f"{base_prompt}\n\n### Input (OCR Text)\n" + "\n\n".join(blocks)
    
    @staticmethod
    def build_ax_prompt(ax_data: List[dict], base_prompt: str) -> str:
        """Build a prompt with accessibility data.
        
        Args:
            ax_data: List of accessibility dictionaries
            base_prompt: Base prompt template
            
        Returns:
            Formatted prompt string
        """
        prompt = base_prompt
        for idx, ax in enumerate(ax_data):
            prompt += f"\n\nAccessibility record {idx}: {ax}"
        return prompt
    
    @staticmethod
    def build_metadata_prompt(metadata_list: List[dict], base_prompt: str) -> str:
        """Build a prompt with metadata records.
        
        Args:
            metadata_list: List of metadata dictionaries
            base_prompt: Base prompt template
            
        Returns:
            Formatted prompt string
        """
        prompt = base_prompt
        for idx, metadata in enumerate(metadata_list):
            prompt += f"\n\nMetadata record {idx}: {metadata}"
        return prompt