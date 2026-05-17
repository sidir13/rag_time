"""Moteur RAG — Retrieval-Augmented Generation
=============================================

Pipeline :
  1. Embedding de la requête (BGE-M3, même modèle que l'ingestion)
  2. Recherche vectorielle + filtres métadonnées (FAISS + post-filtrage)
  3. Construction du prompt avec contexte (top-k chunks retrouvés)
  4. Appel LLM GPT-4o via OpenRouter (API compatible OpenAI)
  5. Retour de la réponse avec sources et scores de similarité

Mode dégradé :
  Si OPENROUTER_API_KEY n'est pas configurée, le moteur retourne
  les chunks retrouvés sans génération LLM (retrieval-only).
  L'application reste fonctionnelle pour l'exploration de tickets.

Note sur l'absence de BM25 :
  L'architecture v1 prévoit un index BM25 (OpenSearch) pour la recherche
  hybride. Pour le MVP Streamlit avec FAISS, on s'appuie uniquement sur
  la recherche vectorielle dense. BGE-M3 capture déjà les correspondances
  lexicales grâce à son architecture BERT dense, ce qui compense partiellement
  l'absence de BM25. L'ajout d'OpenSearch est prévu pour la v2.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from embedder import OpenAIEmbedder
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = ROOT / "data" / "vectorstore"
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TOP_K = int(os.getenv("TOP_K", "5"))

SYSTEM_PROMPT = """\
Tu es un assistant expert en support technique. Tu analyses les tickets \
de support client pour aider les agents à résoudre les problèmes.

Règles :
1. Réponds UNIQUEMENT à partir du contexte fourni (les tickets retrouvés).
2. Si le contexte ne contient pas d'information pertinente, dis-le clairement.
3. Cite les ticket_id pertinents dans ta réponse.
4. Structure ta réponse : diagnostic probable, solution proposée, tickets similaires.
5. Réponds dans la langue de la question.
"""


class RAGEngine:
    """Moteur RAG : retrieval → prompt → LLM → réponse avec sources."""

    def __init__(
        self,
        model: OpenAIEmbedder | None = None,
        index: faiss.Index | None = None,
        store: dict | None = None,
        embed_model: str = EMBED_MODEL,
        persist_dir: str | Path = VECTORSTORE_DIR,
        llm_model: str = LLM_MODEL,
        top_k: int = TOP_K,
    ):
        self.top_k = top_k
        self.llm_model = llm_model
        self.persist_dir = Path(persist_dir)

        # Embedding (même modèle que l'ingestion — obligatoire pour la cohérence)
        if model is not None:
            self.model = model
        else:
            self.model = OpenAIEmbedder(embed_model)

        # FAISS index + metadata store (partagé avec Ingestor ou chargé du disque)
        if index is not None and store is not None:
            self.index = index
            self.store = store
        else:
            idx_path = self.persist_dir / "index.faiss"
            meta_path = self.persist_dir / "store.pkl"
            if idx_path.exists() and meta_path.exists():
                self.index = faiss.read_index(str(idx_path))
                with open(meta_path, "rb") as f:
                    self.store = pickle.load(f)
            else:
                dim = self.model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatIP(dim)
                self.store = {"ids": [], "documents": [], "metadatas": []}

        # Client OpenRouter (API compatible OpenAI — mode retrieval-only si absent)
        self.llm = None
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            from openai import OpenAI

            self.llm = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

    @property
    def index_count(self) -> int:
        """Nombre de documents dans l'index."""
        return self.index.ntotal

    @property
    def has_llm(self) -> bool:
        return self.llm is not None

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> dict:
        """Recherche les chunks les plus pertinents via FAISS.

        Stratégie : over-retrieve (top_k * 5) puis post-filtrage métadonnées.
        Avec IndexFlatIP + vecteurs L2-normalisés, les scores retournés
        sont directement la similarité cosinus (plus haut = plus pertinent).

        Args:
            query: question utilisateur
            top_k: nombre de résultats (défaut: self.top_k)
            filters: dict de filtres metadata {champ: valeur}

        Returns:
            dict avec documents, metadatas, similarities
        """
        if self.index_count == 0:
            return {"documents": [], "metadatas": [], "similarities": []}

        k = top_k or self.top_k
        # Over-retrieve pour compenser le post-filtrage
        fetch_k = min(k * 5, self.index_count) if filters else min(k, self.index_count)

        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        )
        D, I = self.index.search(
            np.array(query_vec, dtype=np.float32), fetch_k
        )

        results: dict = {"documents": [], "metadatas": [], "similarities": []}
        for sim, idx in zip(D[0], I[0]):
            if idx < 0:  # FAISS retourne -1 pour les résultats vides
                continue
            meta = self.store["metadatas"][idx]
            if filters and not self._matches_filters(meta, filters):
                continue
            results["documents"].append(self.store["documents"][idx])
            results["metadatas"].append(meta)
            results["similarities"].append(float(sim))
            if len(results["documents"]) >= k:
                break

        return results

    @staticmethod
    def _matches_filters(meta: dict, filters: dict) -> bool:
        """Vérifie qu'un document correspond aux filtres métadonnées."""
        for key, value in filters.items():
            if value and str(value) != "Tous":
                if meta.get(key) != str(value):
                    return False
        return True

    # ── Formatage du contexte ────────────────────────────────────────────────

    def build_context(self, results: dict) -> str:
        """Formate les résultats de retrieval en contexte lisible pour le LLM."""
        if not results["documents"]:
            return "Aucun ticket pertinent trouvé dans la base."

        parts = []
        for i, (doc, meta, sim) in enumerate(
            zip(
                results["documents"],
                results["metadatas"],
                results["similarities"],
            )
        ):
            similarity = max(0.0, round(sim, 4))
            ticket_id = meta.get("ticket_id", meta.get("filename", "inconnu"))
            source = meta.get("source", "inconnu")

            header = (
                f"[Ticket #{i + 1} | ID: {ticket_id} | "
                f"Similarité: {similarity:.1%} | Source: {source}]"
            )

            meta_display = []
            for field in [
                "product", "category", "priority", "status", "language", "region"
            ]:
                if field in meta:
                    meta_display.append(f"{field}={meta[field]}")
            if meta_display:
                header += f"\n  Métadonnées: {', '.join(meta_display)}"

            parts.append(f"{header}\n{doc}")

        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, results: dict) -> list[dict]:
        """Extrait les sources structurées pour l'affichage frontend."""
        sources = []
        if not results["documents"]:
            return sources

        for doc, meta, sim in zip(
            results["documents"],
            results["metadatas"],
            results["similarities"],
        ):
            similarity = max(0.0, round(sim, 4))
            sources.append(
                {
                    "ticket_id": meta.get(
                        "ticket_id", meta.get("filename", "?")
                    ),
                    "source_type": meta.get("source", "?"),
                    "similarity": similarity,
                    "product": meta.get("product", ""),
                    "category": meta.get("category", ""),
                    "language": meta.get("language", ""),
                    "status": meta.get("status", ""),
                    "priority": meta.get("priority", ""),
                    "region": meta.get("region", ""),
                    "excerpt": doc[:300] + "…" if len(doc) > 300 else doc,
                    "full_text": doc,
                }
            )
        return sources

    # ── Pipeline RAG complet ─────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
        temperature: float = 0.3,
    ) -> dict:
        """Pipeline RAG complet : retrieval → prompt → LLM → réponse.

        Args:
            question: question utilisateur
            top_k: nombre de résultats à retrouver
            filters: filtres métadonnées optionnels
            temperature: température du LLM (0.0 = déterministe)

        Returns:
            dict avec answer, sources, query, filters_applied, num_results
        """
        # 1. Retrieval
        results = self.retrieve(question, top_k=top_k, filters=filters)
        context = self.build_context(results)
        sources = self._extract_sources(results)

        # 2. Génération (ou mode dégradé)
        if self.llm is not None:
            answer = self._generate(context, question, temperature)
        else:
            answer = (
                "**Mode retrieval uniquement** "
                "(OPENROUTER_API_KEY non configurée)\n\n"
                f"{context}"
            )

        return {
            "answer": answer,
            "sources": sources,
            "query": question,
            "filters_applied": filters,
            "num_results": len(sources),
        }

    def _generate(self, context: str, question: str, temperature: float) -> str:
        """Appel au LLM GPT-4o via OpenRouter."""
        user_prompt = (
            f"Contexte (tickets de support retrouvés) :\n{context}\n\n"
            f"Question de l'utilisateur :\n{question}\n\n"
            "Réponds en te basant uniquement sur les tickets ci-dessus."
        )
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                max_tokens=2048,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return (
                f"Erreur lors de l'appel à l'API OpenRouter : {e}\n\n"
                f"**Contexte retrouvé (mode dégradé) :**\n\n{context}"
            )
