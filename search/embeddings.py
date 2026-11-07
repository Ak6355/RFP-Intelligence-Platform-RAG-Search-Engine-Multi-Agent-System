"""Local BGE embedding wrapper used by the RFP indexing pipeline."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """Loads BGE and generates normalized passage/query vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        if self.dimension <= 0:
            raise RuntimeError("Could not determine embedding dimension from model")

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Encode document chunks. BGE instructions are not added to passages."""
        clean_texts = [str(text or "").strip() for text in texts]
        if not clean_texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            clean_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a short search query using BGE's recommended query instruction."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        vector = self.model.encode(
            [QUERY_INSTRUCTION + query.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return np.asarray(vector, dtype=np.float32)
