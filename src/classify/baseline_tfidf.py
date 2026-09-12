import pickle
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

def train_baselines(train_path: str = "data/golden/golden_set.csv"):
    df = pd.read_csv(train_path)
    X = df["customer_text"]
    y = df["intent"]

    # Baseline 1: Trivial Majority Class Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X, y)
    print("--- Baseline 1: Majority Class ---")
    print(classification_report(y, dummy.predict(X), zero_division=0))

    # Baseline 2: TF-IDF + Logistic Regression
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_vec, y)

    print("\n--- Baseline 2: TF-IDF + Logistic Regression ---")
    print(classification_report(y, clf.predict(X_vec), zero_division=0))

    # Save artifacts
    with open("data/processed/tfidf_baseline.pkl", "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "classifier": clf, "dummy": dummy}, f)
    print("Saved baseline models to data/processed/tfidf_baseline.pkl")

if __name__ == "__main__":
    train_baselines()