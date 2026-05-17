import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from nomic import embed
from sentence_transformers import SentenceTransformer



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

    
    

    # =========================
    # 3. Load model (LOCAL)
    # =========================
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="cuda"  # 🔥 this is key
    )
    



    # =========================
    # 4. Generate embeddings
    # =========================
    BATCH_SIZE = 256
    embeddings = []

    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(
        range(0, len(df), BATCH_SIZE),
        total=num_batches,
        desc="Embedding batches",
        unit="batch"
    ):
        batch = df["text"].iloc[i:i+BATCH_SIZE].tolist()
        
        batch = [f"search_document: {t}" for t in batch]
        
        batch_embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False  # keep this False to avoid nested bars
        )
        
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings).astype("float32")

    print("Embedding shape:", embeddings.shape)