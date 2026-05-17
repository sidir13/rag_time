import faiss

def inxa_fass(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)  # 👈 changed from L2
    index.add(embeddings)

    print("Total vectors in index:", index.ntotal)

    # =========================
    # 5. Save index + metadata
    # =========================
    faiss.write_index(index, "tickets_index.faiss")
    
    