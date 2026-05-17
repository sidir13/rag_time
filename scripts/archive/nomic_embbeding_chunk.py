import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from nomic import embed
from sentence_transformers import SentenceTransformer

def create_chunks(row):
    chunks = []

    chunks.append(f"Issue: {row['issue_description']}")
    chunks.append(f"Resolution: {row['resolution_notes']}")
    chunks.append(
        f"Context: Product={row['product']}, Category={row['category']}, Priority={row['priority']}"
    )

    return chunks

def nomic_embed(df):
    # =========================
    # 2. Create text to embed
    # (VERY IMPORTANT STEP)
    # =========================
    def combine_fields(row):
        return f"""
        Product: {row['product']}
        Category: {row['category']}
        Issue: {row['issue_description']}
        Notes: {row['resolution_notes']}
        Priority: {row['priority']}
        Status: {row['status']}
        Channel: {row['channel']}
        Region: {row['region']}
        """

    df["text"] = df.apply(combine_fields, axis=1)

    
    
    chunk_texts = []
    chunk_metadata = []

    for _, row in df.iterrows():
        chunks = create_chunks(row)
        
        for chunk in chunks:
            chunk_texts.append(chunk)
            chunk_metadata.append({
                "ticket_id": row["ticket_id"],
                "type": chunk.split(":")[0],  # Issue / Resolution / Context
                "original_text": row["text"]
            })
    # =========================
    # 3. Load model (LOCAL)
    # =========================
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="cuda"  # 🔥 this is key
    )
    # Move model to GPU (if not already)
    model = model.to("cuda")

    embeddings = model.encode(
        [f"search_document: {t}" for t in chunk_texts],
        batch_size=512,              # 🚀 increase for GPU
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        device="cuda"                # 🔥 force GPU
    )
    np.save("embeddings_chunks.npy", embeddings)
    






    