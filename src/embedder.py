"""Adaptateur d'embedding OpenAI — remplace SentenceTransformer.

Offre la même interface que SentenceTransformer (encode, get_sentence_embedding_dimension)
pour permettre une substitution transparente dans RAGEngine et Ingestor.

Modèles supportés :
  text-embedding-3-small  : 1536 dims, ~20× moins cher qu'ada-002
  text-embedding-3-large  : 3072 dims, meilleure qualité
  text-embedding-ada-002  : 1536 dims (legacy)

Clé API : variable d'env OPENAI_API_KEY
"""

import os
from typing import List

import numpy as np


class OpenAIEmbedder:
    """Wrapper OpenAI Embeddings API compatible SentenceTransformer.

    Gère le batching automatique pour respecter les limites de l'API OpenAI
    (max 2048 inputs par requête pour text-embedding-3-small).
    """

    _DIMS = {
        "text-embedding-3-small":  1536,
        "text-embedding-3-large":  3072,
        "text-embedding-ada-002":  1536,
    }

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it or add it to your .env file."
            )
        self.model_name = model
        self._client = OpenAI(api_key=api_key)
        self._dim = self._DIMS.get(model, 1536)

    def get_sentence_embedding_dimension(self) -> int:
        """Retourne la dimension des vecteurs produits."""
        return self._dim

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,  # no-op: OpenAI embeddings are already L2-norm
        show_progress_bar: bool = False,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Encode une liste de textes via l'API OpenAI.

        Args:
            texts: liste de chaînes à encoder
            normalize_embeddings: ignoré (OpenAI renvoie déjà des vecteurs normalisés)
            show_progress_bar: ignoré (pas de barre locale pour les appels API)
            batch_size: taille des batchs envoyés à l'API (≤ 2048)

        Returns:
            np.ndarray de shape (len(texts), dim), dtype float32
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=self.model_name,
            )
            # L'API peut retourner les items dans n'importe quel ordre
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        return np.array(all_embeddings, dtype=np.float32)
