"""Rebuild FAISS vectorstore using a local SentenceTransformer model.

Re-embeds every document already stored in the existing store.pkl
(no need to re-parse the original CSV) and writes a fresh index.faiss.

Usage:
    python scripts/rebuild_vectorstore.py [--src PATH] [--dst PATH] [--model MODEL]

Defaults:
    --src   data/vectorstore_multilingual_test   (existing store)
    --dst   data/vectorstore_multilingual_test   (overwrite in-place)
    --model paraphrase-multilingual-MiniLM-L12-v2

No API key required — model is downloaded once and cached locally.
"""

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np
from tqdm import tqdm

from embedder import LocalEmbedder


def rebuild(src_dir: Path, dst_dir: Path, model_name: str, batch_size: int) -> None:
    # ── 1. Load existing store ───────────────────────────────────────────────
    store_path = src_dir / "store.pkl"
    if not store_path.exists():
        raise FileNotFoundError(f"store.pkl not found in {src_dir}")

    print(f"Loading store from {store_path} …")
    with open(store_path, "rb") as f:
        store: dict = pickle.load(f)

    documents: list[str] = store["documents"]
    total = len(documents)
    print(f"  {total:,} documents to re-embed")

    # ── 2. Init embedder ─────────────────────────────────────────────────────
    embedder = LocalEmbedder(model_name)
    dim = embedder.get_sentence_embedding_dimension()
    print(f"  Model : {model_name}  |  dim : {dim}")

    # ── 3. Embed in batches ──────────────────────────────────────────────────
    all_embeddings: list[np.ndarray] = []

    batches = range(0, total, batch_size)
    for start in tqdm(batches, desc="Embedding", unit="batch"):
        batch = documents[start : start + batch_size]
        vecs = embedder.encode(batch, normalize_embeddings=True)
        all_embeddings.append(vecs)

    matrix = np.vstack(all_embeddings).astype(np.float32)
    assert matrix.shape == (total, dim), f"Shape mismatch: {matrix.shape}"
    print(f"  Embedding matrix: {matrix.shape}")

    # ── 4. Build new FAISS index ─────────────────────────────────────────────
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    print(f"  FAISS index: {index.ntotal:,} vectors")

    # ── 5. Persist ───────────────────────────────────────────────────────────
    dst_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(dst_dir / "index.faiss"))
    with open(dst_dir / "store.pkl", "wb") as f:
        pickle.dump(store, f)

    print(f"\nDone — index saved to {dst_dir}")
    print("  index.faiss  +  store.pkl  (metadata unchanged)")


def main() -> None:
    default_vs = str(ROOT / "data" / "vectorstore_multilingual_test")

    parser = argparse.ArgumentParser(description="Rebuild FAISS vectorstore with OpenAI embeddings")
    parser.add_argument("--src",   default=default_vs, help="Source vectorstore directory")
    parser.add_argument("--dst",   default=default_vs, help="Destination directory (default: overwrite src)")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="HF model ID")
    parser.add_argument("--batch-size", type=int, default=512, help="API batch size (≤ 2048)")
    args = parser.parse_args()

    rebuild(
        src_dir    = Path(args.src),
        dst_dir    = Path(args.dst),
        model_name = args.model,
        batch_size = args.batch_size,
    )


if __name__ == "__main__":
    main()
