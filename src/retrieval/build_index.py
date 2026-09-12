import faiss
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

def build_faiss_index(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    parquet_path = cfg["paths"]["threads_parquet"]
    index_out = cfg["retrieval"]["index_path"]
    meta_out = cfg["retrieval"]["metadata_path"]
    model_name = cfg["models"]["embedding"]

    print(f"Building retrieval index from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # De-duplicate queries
    df = df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)

    print(f"Loading embedding model: {model_name}...")
    encoder = SentenceTransformer(model_name)

    # Encode customer queries as index keys
    embeddings = encoder.encode(
        df["customer_text"].tolist(),
        show_progress_bar=True,
        batch_size=128,
        normalize_embeddings=True # Cosine similarity via Inner Product
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    Path(index_out).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_out)

    # Save metadata aligned with row IDs
    df[["customer_tweet_id", "customer_text", "brand_text"]].to_parquet(meta_out, index=True)
    print(f"FAISS index with {index.ntotal} records written to {index_out}")
    print(f"Metadata table written to {meta_out}")

if __name__ == "__main__":
    build_faiss_index()