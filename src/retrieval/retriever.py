import faiss
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class SupportRetriever:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.encoder = SentenceTransformer(self.cfg["models"]["embedding"])
        self.index = faiss.read_index(self.cfg["retrieval"]["index_path"])
        self.metadata = pd.read_parquet(self.cfg["retrieval"]["metadata_path"])
        self.top_k = self.cfg["retrieval"]["top_k"]

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        query_vec = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx]
            results.append({
                "similarity_score": float(score),
                "customer_text": row["customer_text"],
                "historical_reply": row["brand_text"],
                "tweet_id": row["customer_tweet_id"]
            })
        return results