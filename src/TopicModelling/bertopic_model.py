"""
Step: Run BERTopic on issue summary+description to determine topics
(the second algorithm required for RQ1).

Built as the explicit 6-step pipeline from the lecture slide
(Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF
procedure" - https://maartengr.github.io/BERTopic/algorithm/algorithm.html)
rather than BERTopic's plain defaults:

  1. Embeddings           - SentenceTransformer('all-MiniLM-L6-v2')
  2. Dimensionality        - UMAP
  3. Clustering            - HDBSCAN
  4. Tokenize topics       - CountVectorizer
  5. Topic representation  - ClassTfidfTransformer (c-TF-IDF)
  6. Fine-tune topic words - KeyBERTInspired

Step 6 is the main upgrade over letting BERTopic use its defaults: it
re-ranks each topic's keywords by how well they represent the topic's
documents semantically, instead of relying on c-TF-IDF word counts alone -
this tends to produce noticeably cleaner top-word lists.

Install first:
    pip install bertopic sentence-transformers umap-learn hdbscan

Usage:
    python bertopic_model.py
"""
import ast

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from topic_config import (
    BERTOPIC_DIR,
    BERTOPIC_EMBEDDING_MODEL,
    BERTOPIC_HDBSCAN_PARAMS,
    BERTOPIC_UMAP_PARAMS,
    BERTOPIC_VECTORIZER_PARAMS,
    EXTRA_STOPWORDS,
    PROCESSED_ISSUES_CSV,
)


def main():
    BERTOPIC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PROCESSED_ISSUES_CSV}...")
    df = pd.read_csv(PROCESSED_ISSUES_CSV)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["tokens"] = df["tokens"].apply(
        lambda toks: [t for t in toks if t not in EXTRA_STOPWORDS]
    )
    # BERTopic wants readable text, not a bag-of-stems - re-join tokens with
    # spaces (already cleaned/lowercased from week 1) rather than raw summary
    # + description, so results are comparable to the LDA vocabulary.
    docs = df["tokens"].apply(lambda t: " ".join(t)).tolist()

    # --- Step 1: Extract embeddings ---
    print(f"Step 1/6: loading embedding model '{BERTOPIC_EMBEDDING_MODEL}'...")
    embedding_model = SentenceTransformer(BERTOPIC_EMBEDDING_MODEL)

    # --- Step 2: Reduce dimensionality ---
    umap_model = UMAP(**BERTOPIC_UMAP_PARAMS)

    # --- Step 3: Cluster reduced embeddings ---
    hdbscan_model = HDBSCAN(**BERTOPIC_HDBSCAN_PARAMS)

    # --- Step 4: Tokenize topics ---
    vectorizer_model = CountVectorizer(**BERTOPIC_VECTORIZER_PARAMS)

    # --- Step 5: Create topic representation (c-TF-IDF) ---
    ctfidf_model = ClassTfidfTransformer()

    # --- Step 6: Fine-tune topic representations with KeyBERTInspired ---
    representation_model = KeyBERTInspired()

    # --- All steps together ---
    topic_model = BERTopic(
        embedding_model=embedding_model,             # Step 1
        umap_model=umap_model,                       # Step 2
        hdbscan_model=hdbscan_model,                 # Step 3
        vectorizer_model=vectorizer_model,            # Step 4
        ctfidf_model=ctfidf_model,                    # Step 5
        representation_model=representation_model,   # Step 6
        calculate_probabilities=False,
        verbose=True,
    )

    print("\nFitting BERTopic (embeds -> reduces -> clusters -> represents)...")
    topics, _ = topic_model.fit_transform(docs)

    df["bertopic_topic"] = topics
    df[["issue_key", "bertopic_topic"]].to_csv(
        BERTOPIC_DIR / "doc_topics.csv", index=False
    )

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(BERTOPIC_DIR / "topic_info.csv", index=False)

    n_outliers = (df["bertopic_topic"] == -1).sum()
    print(f"\n{len(topic_info) - 1} topics found "
          f"(+ {n_outliers} outlier issues, {n_outliers / len(df):.0%} of corpus)")
    print("\nTopic overview (topic -1 = outliers, not assigned to any topic):")
    print(topic_info[["Topic", "Count", "Name"]].head(25).to_string(index=False))

    topic_model.save(str(BERTOPIC_DIR / "bertopic_model"), serialization="safetensors")
    print(f"\nSaved: {BERTOPIC_DIR / 'doc_topics.csv'}")
    print(f"Saved: {BERTOPIC_DIR / 'topic_info.csv'}")
    print(f"Saved model to: {BERTOPIC_DIR / 'bertopic_model'}")


if __name__ == "__main__":
    main()