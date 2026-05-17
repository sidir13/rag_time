"""Script de ré-indexation — migration vers nomic-embed-text-v1.5 + 3 chunks/ticket
=====================================================================================

Ce script :
  1. Supprime l'ancien index FAISS (BAAI/bge-m3, 1024 dims, 1 chunk/ticket)
  2. Reconstruit l'index avec nomic-ai/nomic-embed-text-v1.5 (768 dims, 3 chunks/ticket)

Usage :
    python reindex.py [--max-rows N] [--batch-size 256]

Options :
    --max-rows    : limiter le nombre de tickets (utile en dev, défaut : tout)
    --batch-size  : taille des batchs d'embedding (défaut : 256)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest import Ingestor, DATA_RAW, VECTORSTORE_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ré-indexation complète (nomic-embed + 3 chunks)")
    parser.add_argument("--max-rows", type=int, default=None, help="Nombre max de tickets")
    parser.add_argument("--batch-size", type=int, default=256, help="Taille des batchs")
    args = parser.parse_args()

    print("=" * 60)
    print("  REINDEXATION — nomic-ai/nomic-embed-text-v1.5")
    print("=" * 60)

    # ── 1. Initialiser l'Ingestor (charge le modèle) ────────────────────────
    ingestor = Ingestor()

    # ── 2. Vider l'ancien index ──────────────────────────────────────────────
    if ingestor.count > 0:
        print(f"\n  Ancien index : {ingestor.count:,} vecteurs — suppression...")
        ingestor.clear()
    else:
        print("\n  Index vide — pas de suppression nécessaire")

    # ── 3. Ré-indexer le CSV ─────────────────────────────────────────────────
    print(f"\n  Ingestion de : {DATA_RAW}")
    if args.max_rows:
        print(f"  Limite : {args.max_rows:,} tickets")

    n_tickets = ingestor.ingest_csv(
        csv_path=DATA_RAW,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )

    print(f"\n  ✓ {n_tickets:,} tickets → {ingestor.count:,} chunks dans l'index")
    print(f"  ✓ Index sauvegardé dans : {VECTORSTORE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
