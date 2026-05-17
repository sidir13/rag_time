"""Adaptateur d embedding via HuggingFace Inference API (serverless).

Zero RAM pour les poids du modele -- tout passe par des appels HTTP.
Modele par defaut : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - 384 dims, 50+ langues (EN, DE, FR, ES, ZH, ...)
  - Plan gratuit HF : ~1 000 req/jour

Env var requise : HF_API_KEY  (huggingface.co/settings/tokens)
"""

import os
from typing import List

import numpy as np


class HFApiEmbedder:
    """Appelle l API d inference HuggingFace pour produire des embeddings.

    Interface identique a SentenceTransformer (encode, get_sentence_embedding_dimension)
    pour etre interchangeable dans RAGEngine et Ingestor.
    """

    _MODEL_DIMS: dict = {
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        "sentence-transformers/all-MiniLM-L6-v2":                       384,
        "sentence-transformers/all-mpnet-base-v2":                      768,
        "BAAI/bge-m3":                                                  1024,
    }

    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        from huggingface_hub import InferenceClient

        api_key = os.environ.get("HF_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "HF_API_KEY est absent. "
                "Ajoute-le dans .env ou dans les variables d env Render."
            )

        self._client = InferenceClient(token=api_key)
        self.model_name = model
        self._dim = self._MODEL_DIMS.get(model, 384)

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_vecs.append(self._embed_batch(batch))
        result = np.vstack(all_vecs).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result /= np.maximum(norms, 1e-9)
        return result

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        result = self._client.feature_extraction(batch, model=self.model_name)
        arr = np.array(result, dtype=np.float32)
        if arr.ndim == 3:   # (batch, seq_len, dim) -> mean pool
            arr = arr.mean(axis=1)
        return arr


# Alias retro-compatibles
LocalEmbedder  = HFApiEmbedder
OpenAIEmbedder = HFApiEmbedder
