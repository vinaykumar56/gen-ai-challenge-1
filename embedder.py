import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name: str='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} loaded successfully.")    

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        Args:
            texts (List[str]): List of texts to be embedded.
        Returns:
            np.ndarray: Array of embeddings."""
        if not self.model:
            raise ValueError("Model is not loaded. Call _load_model() first.")
    
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    