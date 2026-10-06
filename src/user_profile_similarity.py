import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
from collections import defaultdict
from typing import Dict, List, Tuple

from src.models.utils.device import resolve_torch_device


class UserProfileSimilarity:
    """
    Calculate similarity between user profiles using multiple ML metrics.
    Uses sentence transformers for semantic understanding.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str | None = "auto"):
        """
        Initialize with sentence transformer model.
        
        Args:
            model_name: HuggingFace model name. Options:
                - 'all-MiniLM-L6-v2' (default, fast, good quality)
                - 'all-mpnet-base-v2' (slower, better quality)
                - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
        """
        self.device = resolve_torch_device(device)
        print(f"Loading sentence transformer model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("Model loaded successfully!")
        
    def flatten_profile(self, profile: Dict) -> Dict[str, List[str]]:
        """Extract all list fields from nested profile structure."""
        flattened = defaultdict(list)
        
        def extract_lists(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    extract_lists(value, new_prefix)
            elif isinstance(data, list):
                # Only store non-empty lists with string values
                if data and all(isinstance(x, str) for x in data):
                    flattened[prefix].extend(data)
        
        extract_lists(profile)
        return flattened
    
    def get_all_text(self, profile: Dict) -> str:
        """Combine all text from profile into single string."""
        flattened = self.flatten_profile(profile)
        all_items = []
        for items in flattened.values():
            all_items.extend(items)
        return " ".join(all_items)
    
    def jaccard_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate Jaccard similarity between two lists."""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
        
        set1 = set(item.lower() for item in list1)
        set2 = set(item.lower() for item in list2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if not text1 or not text2:
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.model.encode([text1, text2])
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def field_semantic_similarity(self, profile1: Dict, profile2: Dict) -> Dict[str, float]:
        """Calculate semantic similarity for each field independently."""
        flat1 = self.flatten_profile(profile1)
        flat2 = self.flatten_profile(profile2)
        
        similarities = {}
        all_keys = set(flat1.keys()).union(set(flat2.keys()))
        
        for key in all_keys:
            text1 = " ".join(flat1.get(key, []))
            text2 = " ".join(flat2.get(key, []))
            
            if text1 and text2:
                similarities[key] = self.semantic_similarity(text1, text2)
            elif not text1 and not text2:
                similarities[key] = 1.0
            else:
                similarities[key] = 0.0
        
        return similarities
    
    def field_wise_similarity(self, profile1: Dict, profile2: Dict) -> Dict[str, float]:
        """Calculate Jaccard similarity for each matching field."""
        flat1 = self.flatten_profile(profile1)
        flat2 = self.flatten_profile(profile2)
        
        similarities = {}
        all_keys = set(flat1.keys()).union(set(flat2.keys()))
        
        for key in all_keys:
            similarities[key] = self.jaccard_similarity(
                flat1.get(key, []), 
                flat2.get(key, [])
            )
        
        return similarities
    
    def weighted_similarity(self, profile1: Dict, profile2: Dict, 
                          weights: Dict[str, float] = None) -> float:
        """
        Calculate weighted average of field similarities.
        
        Default weights prioritize:
        - Goals and skills (high importance)
        - Interests and preferences (medium importance)
        - Demographics (low importance)
        """
        if weights is None:
            weights = {
                'goals': 0.25,
                'skills': 0.20,
                'interests': 0.20,
                'knowledge': 0.15,
                'preferences': 0.10,
                'background': 0.05,
                'traits': 0.05
            }
        
        field_sims = self.field_wise_similarity(profile1, profile2)
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for field, similarity in field_sims.items():
            # Match field to weight category
            weight = 0.05  # default
            for category, w in weights.items():
                if category in field.lower():
                    weight = w
                    break
            
            weighted_sum += similarity * weight
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def calculate_similarity(self, profile1: Dict, profile2: Dict) -> Dict[str, any]:
        """
        Calculate comprehensive similarity metrics between two profiles.
        
        Returns:
            Dictionary with multiple similarity scores and detailed breakdown
        """
        # Overall semantic similarity
        text1 = self.get_all_text(profile1)
        text2 = self.get_all_text(profile2)
        semantic_sim = self.semantic_similarity(text1, text2)
        
        field_jaccard = self.field_wise_similarity(profile1, profile2)
        field_semantic = self.field_semantic_similarity(profile1, profile2)
        
        weighted_sim = self.weighted_similarity(profile1, profile2)
        
        avg_jaccard = np.mean(list(field_jaccard.values())) if field_jaccard else 0.0
        avg_semantic = np.mean(list(field_semantic.values())) if field_semantic else 0.0
    
        composite_score = (
            0.5 * semantic_sim +
            0.3 * weighted_sim +
            0.2 * avg_jaccard
        )
        
        return {
            'composite_score': round(composite_score, 4),
            'semantic_similarity': round(semantic_sim, 4),
            'weighted_jaccard': round(weighted_sim, 4),
            'average_jaccard': round(avg_jaccard, 4),
            'average_semantic': round(avg_semantic, 4),
            'field_jaccard': {k: round(v, 4) for k, v in field_jaccard.items()},
            'field_semantic': {k: round(v, 4) for k, v in field_semantic.items()},
            'interpretation': self._interpret_score(composite_score)
        }
    
    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of similarity score."""
        if score >= 0.8:
            return "Very High Similarity"
        elif score >= 0.6:
            return "High Similarity"
        elif score >= 0.4:
            return "Moderate Similarity"
        elif score >= 0.2:
            return "Low Similarity"
        else:
            return "Very Low Similarity"
    
    def compare_multiple_profiles(self, profiles: List[Dict]) -> np.ndarray:
        """
        Create similarity matrix for multiple profiles.
        
        Args:
            profiles: List of user profile dictionaries
            
        Returns:
            NxN similarity matrix where N is number of profiles
        """
        n = len(profiles)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    result = self.calculate_similarity(profiles[i], profiles[j])
                    sim = result['composite_score']
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
        
        return similarity_matrix


# Example usage
if __name__ == "__main__":
    
    with open("../output_data/anastiasia/screen_qwen3_vl_4b_instruct_8bit/user_profile.json", "r", encoding="utf-8") as f:
        profile1 = json.load(f)
    
    with open("../output_data/anastiasia/ax_qwen3_vl_4b_instruct_8bit/user_profile.json", "r", encoding="utf-8") as f:
        profile2 = json.load(f)

    calculator = UserProfileSimilarity()
    
    result = calculator.calculate_similarity(profile1, profile2)
    
    print(f"\nOverall Metrics:")
    print(f"  • Composite Score: {result['composite_score']} ({result['interpretation']})")
    print(f"  • Semantic Similarity (Transformers): {result['semantic_similarity']}")
    print(f"  • Weighted Jaccard: {result['weighted_jaccard']}")
    print(f"  • Average Jaccard: {result['average_jaccard']}")
    print(f"  • Average Semantic: {result['average_semantic']}")
    
    print(f"\n📋 Field-by-Field Semantic Similarity:")
    for field, score in sorted(result['field_semantic'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"  • {field}: {score}")
    
    print(f"\n📋 Field-by-Field Jaccard Similarity:")
    for field, score in sorted(result['field_jaccard'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"  • {field}: {score}")
    
    # Example with multiple profiles
    # print(f"\n\n{'=' * 60}")
    # print("SIMILARITY MATRIX (Multiple Profiles)")
    # print("=" * 60)
    
    # profiles = [profile1, profile2]
    # matrix = calculator.compare_multiple_profiles(profiles)
    
    # print("\nSimilarity Matrix:")
    # print(matrix)
    # print("\nNote: Values range from 0 (no similarity) to 1 (identical profiles)")
