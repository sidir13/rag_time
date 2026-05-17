"""Pipeline d'ingestion RAG — Tickets de support + Documents
============================================================

Architecture (cf. architecture_rag_v1.md) :
- 1 ticket CSV = 1 chunk (~80-120 tokens)
- Documents uploadés (PDF/TXT/MD) : chunking configurable (taille + overlap)
- Embedding : BAAI/bge-m3 (1024 dims, multilingue — 100+ langues)
- Vector store : FAISS IndexFlatIP (persistant sur disque)

Choix FAISS vs ChromaDB :
  FAISS (IndexFlatIP) offre une recherche exacte par produit scalaire sur des
  vecteurs L2-normalisés (équivalent cosine). Pour 200k vecteurs × 1024 dims,
  la recherche prend ~1 ms et la mémoire ~800 Mo.
  ChromaDB offrirait un filtrage natif mais nécessite la compilation C++ de
  hnswlib (MSVC Build Tools indisponibles sur cet environnement).
  Les métadonnées sont stockées dans un pickle séparé avec post-filtrage.

Choix BGE-M3 vs text-embedding-3-large :
  BGE-M3 couvre les 6 langues du corpus (FR/EN/DE/ES/JA/ZH) avec un seul modèle,
  s'exécute localement (pas de coût API par requête), produit 1024 dims avec un
  contexte de 8192 tokens. text-embedding-3-large nécessite des appels API payants
  et n'est pas optimisé pour le japonais/chinois.
"""

import os
import sys
import hashlib
import pickle
import unicodedata
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 ne supporte pas → ✓ etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import faiss
import numpy as np
import pandas as pd
from embedder import OpenAIEmbedder
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "customer_support_tickets_200k.csv"
VECTORSTORE_DIR = ROOT / "data" / "vectorstore"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

PII_FIELDS = {"customer_name", "customer_email"}

# 13 champs de métadonnées issus de l'EDA (cf. eda_rapport.md)
METADATA_FIELDS = [
    "product", "category", "priority", "status", "channel",
    "region", "language", "operating_system", "subscription_type",
    "customer_segment", "escalated", "sla_breached",
    "ticket_created_date", "ticket_resolved_date",
]


# ── Utilitaires ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalisation unicode NFC — cohérence multilingue."""
    return unicodedata.normalize("NFC", text.strip())


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Découpage d'un texte long en chunks avec overlap.

    Utilisé uniquement pour les documents uploadés (PDF/TXT/MD).
    Les tickets CSV sont 1:1 (pas de segmentation).
    Tente de couper au dernier point ou retour à la ligne pour préserver
    la cohérence sémantique des chunks.
    """
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if end < len(text):
            cut = max(chunk.rfind("."), chunk.rfind("\n"))
            if cut > size * 0.5:
                chunk = chunk[: cut + 1]
                end = start + cut + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


# ── Classe principale ───────────────────────────────────────────────────────

class Ingestor:
    """Pipeline d'ingestion unifié (CSV tickets + documents uploadés).

    Stocke les vecteurs dans FAISS (IndexFlatIP) et les métadonnées dans un
    pickle séparé. Fichiers persistés dans persist_dir :
      - index.faiss  : index FAISS (vecteurs float32)
      - store.pkl    : {ids, documents, metadatas} (pickle)
    """

    def __init__(
        self,
        model: OpenAIEmbedder | None = None,
        index: faiss.Index | None = None,
        store: dict | None = None,
        embed_model: str = EMBED_MODEL,
        persist_dir: str | Path = VECTORSTORE_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "index.faiss"
        self.meta_path = self.persist_dir / "store.pkl"

        # Modèle d'embedding (partageable avec RAGEngine pour éviter le doublon)
        if model is not None:
            self.model = model
        else:
            print(f"Initialisation de l'embedder OpenAI : {embed_model}...")
            self.model = OpenAIEmbedder(embed_model)
            print(f"  → {self.model.get_sentence_embedding_dimension()} dimensions")

        self.dim = self.model.get_sentence_embedding_dimension()

        # Index FAISS + metadata store : partagé ou chargé depuis le disque
        if index is not None and store is not None:
            self.index = index
            self.store = store
        elif self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self.store = pickle.load(f)
        else:
            # IndexFlatIP : produit scalaire sur vecteurs L2-normalisés = cosine
            self.index = faiss.IndexFlatIP(self.dim)
            self.store = {"ids": [], "documents": [], "metadatas": []}

        self._id_set: set[str] = set(self.store["ids"])

    @property
    def count(self) -> int:
        """Nombre de documents dans l'index."""
        return self.index.ntotal

    def save(self):
        """Persiste l'index FAISS et les métadonnées sur disque."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.store, f)

    # ── Ingestion CSV ────────────────────────────────────────────────────────

    def ingest_csv(
        self,
        csv_path: str | Path = DATA_RAW,
        batch_size: int = 256,
        resolved_only: bool = False,
        max_rows: int | None = None,
    ) -> int:
        """Ingère le CSV de tickets de support.

        Args:
            csv_path: chemin du fichier CSV
            batch_size: taille des batchs d'embedding (ajuster selon la RAM GPU/CPU)
            resolved_only: ne garder que les tickets Resolved/Closed
            max_rows: limiter le nombre de lignes (utile en dev/test)

        Returns:
            nombre de tickets indexés
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {csv_path}")

        df = pd.read_csv(csv_path)

        # Suppression PII (customer_name, customer_email)
        df = df.drop(
            columns=[c for c in PII_FIELDS if c in df.columns], errors="ignore"
        )

        if resolved_only:
            df = df[df["status"].isin(["Resolved", "Closed"])]

        if max_rows:
            df = df.head(max_rows)

        total_added = 0
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start : start + batch_size]
            new_ids, new_docs, new_metas = [], [], []

            for row_idx, row in batch_df.iterrows():
                ticket_id = str(row.get("ticket_id", f"ticket_{row_idx}"))

                # Déduplication : skip si déjà indexé (on vérifie le premier chunk)
                if f"{ticket_id}__issue" in self._id_set:
                    continue

                product = str(row.get("product", ""))
                category = str(row.get("category", ""))
                priority = str(row.get("priority", ""))
                issue = _normalize(str(row.get("issue_description", "")))
                resolution = _normalize(str(row.get("resolution_notes", "")))

                # Stratégie 3 chunks par ticket (nomic-embed chunking) :
                #   chunk 1 — Issue        : question/problème du client
                #   chunk 2 — Resolution   : solution appliquée
                #   chunk 3 — Context      : métadonnées structurées
                chunks = [
                    (f"{ticket_id}__issue",      f"Issue: {issue}",      "issue"),
                    (f"{ticket_id}__resolution", f"Resolution: {resolution}", "resolution"),
                    (f"{ticket_id}__context",
                     f"Context: Product={product}, Category={category}, Priority={priority}",
                     "context"),
                ]

                # Métadonnées indexables (13 champs)
                base_meta: dict = {"source": "csv", "ticket_id": ticket_id}
                for field in METADATA_FIELDS:
                    if field in row.index and pd.notna(row[field]):
                        base_meta[field] = str(row[field])

                for chunk_id, chunk_text, chunk_type in chunks:
                    meta = {**base_meta, "chunk_type": chunk_type}
                    new_ids.append(chunk_id)
                    new_docs.append(chunk_text)
                    new_metas.append(meta)

            if not new_ids:
                continue

            # Embedding en batch (OpenAI API), normalisation L2 implicite
            embeddings = self.model.encode(
                new_docs, show_progress_bar=False, normalize_embeddings=True
            )

            self.index.add(np.array(embeddings, dtype=np.float32))
            self.store["ids"].extend(new_ids)
            self.store["documents"].extend(new_docs)
            self.store["metadatas"].extend(new_metas)
            self._id_set.update(new_ids)

            total_added += len(new_ids) // 3  # 3 chunks par ticket
            print(f"  Ingéré {total_added:>6d} / {len(df)} tickets ({self.count} chunks)", end="\r")

        self.save()
        print(f"\n  Terminé : {total_added} nouveaux tickets → {self.count} chunks (total index)")
        return total_added

    # ── Ingestion documents uploadés ─────────────────────────────────────────

    def ingest_document(
        self, file_path: str | Path, content: str | None = None
    ) -> int:
        """Ingère un document (PDF/TXT/MD) avec chunking configurable.

        Args:
            file_path: chemin du fichier
            content: contenu textuel (si déjà extrait, ex: upload Streamlit)

        Returns:
            nombre de chunks indexés
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if content is None:
            if suffix == ".pdf":
                content = self._read_pdf(file_path)
            elif suffix in (".txt", ".md", ".markdown"):
                content = file_path.read_text(encoding="utf-8")
            else:
                raise ValueError(
                    f"Format non supporté : {suffix}. Formats attendus : .pdf, .txt, .md"
                )

        if not content or not content.strip():
            raise ValueError(f"Document vide ou illisible : {file_path.name}")

        content = _normalize(content)
        chunks = _chunk_text(content, self.chunk_size, self.chunk_overlap)

        new_ids, new_docs, new_metas = [], [], []
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(
                f"{file_path.name}_{i}".encode()
            ).hexdigest()[:16]

            if doc_id in self._id_set:
                continue

            new_ids.append(doc_id)
            new_docs.append(chunk)
            new_metas.append(
                {
                    "source": "upload",
                    "filename": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )

        if not new_ids:
            print(f"  {file_path.name} : tous les chunks déjà indexés")
            return 0

        embeddings = self.model.encode(
            new_docs, show_progress_bar=False, normalize_embeddings=True
        )

        self.index.add(np.array(embeddings, dtype=np.float32))
        self.store["ids"].extend(new_ids)
        self.store["documents"].extend(new_docs)
        self.store["metadatas"].extend(new_metas)
        self._id_set.update(new_ids)

        self.save()
        print(f"  {file_path.name} -> {len(new_ids)} chunks indexés")
        return len(new_ids)

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extraction texte d'un PDF via pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf est requis pour les PDF : pip install pypdf")

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        if not pages:
            raise ValueError(
                f"Impossible d'extraire du texte du PDF : {path.name}"
            )
        return "\n\n".join(pages)

    def clear(self):
        """Supprime l'intégralité de l'index (in-place pour garder les refs)."""
        self.index.reset()
        self.store["ids"].clear()
        self.store["documents"].clear()
        self.store["metadatas"].clear()
        self._id_set.clear()
        self.save()
        print("  Index vidé")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion RAG — Tickets de support")
    parser.add_argument("--csv", default=str(DATA_RAW), help="Chemin du CSV")
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limite de lignes (par défaut : tout le CSV)"
    )
    parser.add_argument(
        "--resolved-only", action="store_true",
        help="N'indexer que les tickets Resolved/Closed"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Vider l'index avant l'ingestion"
    )
    args = parser.parse_args()

    ing = Ingestor()
    if args.clear:
        ing.clear()
    ing.ingest_csv(
        csv_path=args.csv,
        max_rows=args.max_rows,
        resolved_only=args.resolved_only,
    )
    print(f"Total dans l'index : {ing.count}")
