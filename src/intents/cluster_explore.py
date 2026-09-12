import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def run_clustering(parquet_path: str, k: int = 10, sample_size: int = 4000):
    print(f"Loading {sample_size} sample records from {parquet_path} for intent mining...")
    df = pd.read_parquet(parquet_path)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    print("Encoding texts with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sample_df["customer_text"].tolist(), show_progress_bar=True, batch_size=64)

    print(f"Clustering into k={k} groups via KMeans...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    sample_df["cluster"] = kmeans.fit_predict(embeddings)

    # Extract cluster keywords via TF-IDF top tokens
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = tfidf.fit_transform(sample_df["customer_text"])
    feature_names = tfidf.get_feature_names_out()

    print("\n--- Intent Cluster Extraction Summary ---")
    for cluster_id in range(k):
        cluster_docs = sample_df[sample_df["cluster"] == cluster_id]
        indices = cluster_docs.index.tolist()
        cluster_tfidf = tfidf_matrix[indices].mean(axis=0).A1
        top_tokens = [feature_names[i] for i in cluster_tfidf.argsort()[-6:][::-1]]
        print(f"\nCluster {cluster_id} (n={len(cluster_docs)}): Keywords: {', '.join(top_tokens)}")
        print("Example tweets:")
        for text in cluster_docs["customer_text"].head(2):
            print(f"  - {text[:110]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/uber_threads.parquet")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run_clustering(args.input, args.k)