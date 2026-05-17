"""Adaptateur d'embedding local — modèle SentenceTransformer léger.

Utilise paraphrase-multilingual-MiniLM-L12-v2 par défaut :
  - Taille : ~118 MB  (téléchargé une seule fois, mis en cache)
  - RAM    : ~150 MB au runtime
  - Dims   : 384
  - Langues : 50+ (EN, DE, FR, ES, ZH, JA, …)
  - Pas de trust_remote_code, pas de einops

Comparé à nomic-ai/nomic-embed-text-v1.5 (768 dims, ~550 MB avec PyTorch)
ce modèle tient largement dans les 512 MB de Render Free.

Aucune clé API requise.
"""

from typing import List

import numpy as np


class LocalEmbedder:
    """Wrapper SentenceTransformer léger, interface compatible avec l'ancien code.

    Expose encode() et get_sentence_embedding_dimension() pour être
    interchangeable avec la précédente implémentation OpenAI.
    """

    def __init__(self, model: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.model_name = model

    def get_sentence_embedding_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 256,
    ) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )


# Alias rétro-compatible (anciens imports OpenAIEmbedder restent valides)
OpenAIEmbedder = LocalEmbedder
