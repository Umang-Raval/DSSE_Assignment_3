"""
Step: Run LDA (Iteration 1: baseline tokens, Iteration 2: ontology-replaced
tokens) and report the top terms per topic + topic proportions per issue.

Usage:
    python lda_model.py baseline     # Iteration 1
    python lda_model.py ontology     # Iteration 2 (needs ontology_mapper.py to have run)
"""
import argparse
import ast
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from topic_config import (
    EXTRA_STOPWORDS,
    ITER1_ALPHA,
    ITER1_BETA,
    ITER1_NUM_TOPICS,
    ITER2_ALPHA,
    ITER2_BETA,
    ITER2_NUM_TOPICS,
    LDA_BASELINE_DIR,
    LDA_ONTOLOGY_DIR,
    MAX_DF,
    MIN_DF,
    PROCESSED_ISSUES_CSV,
    PROCESSED_ONTOLOGY_CSV,
    TOP_N_WORDS,
)


def load_documents(mode: str) -> tuple[pd.DataFrame, list[str]]:
    if mode == "baseline":
        df = pd.read_csv(PROCESSED_ISSUES_CSV)
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
        df["tokens"] = df["tokens"].apply(
            lambda toks: [t for t in toks if t not in EXTRA_STOPWORDS]
        )
        texts = df["tokens"].apply(lambda t: " ".join(t)).tolist()
    elif mode == "ontology":
        df = pd.read_csv(PROCESSED_ONTOLOGY_CSV)
        df["tokens_ontology"] = df["tokens_ontology"].apply(ast.literal_eval)
        texts = df["clean_text_ontology"].tolist()
    else:
        raise ValueError("mode must be 'baseline' or 'ontology'")
    return df, texts


def run_lda(texts: list[str], num_topics: int, alpha: float, beta: float):
    vectorizer = CountVectorizer(min_df=MIN_DF, max_df=MAX_DF)
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        doc_topic_prior=alpha,   # alpha
        topic_word_prior=beta,   # beta / eta
        learning_method="online",
        max_iter=50,
        random_state=42,
        n_jobs=-1,
    )
    doc_topic = lda.fit_transform(dtm)
    return lda, vectorizer, feature_names, doc_topic


def top_words_per_topic(lda, feature_names, n_top_words=TOP_N_WORDS) -> dict[int, list[str]]:
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[: -n_top_words - 1 : -1]
        topics[topic_idx] = [feature_names[i] for i in top_idx]
    return topics


def topic_sizes(topics: dict[int, list[str]], lda) -> dict[int, int]:
    """Number of terms with a meaningfully high weight in each topic
    (used to flag under-developed topics, per the assignment)."""
    sizes = {}
    for topic_idx, topic in enumerate(lda.components_):
        threshold = topic.mean() + topic.std()
        sizes[topic_idx] = int((topic > threshold).sum())
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "ontology"])
    args = parser.parse_args()

    if args.mode == "baseline":
        num_topics, alpha, beta, out_dir = (
            ITER1_NUM_TOPICS, ITER1_ALPHA, ITER1_BETA, LDA_BASELINE_DIR,
        )
        print(f"=== Iteration 1: baseline tokens, {num_topics} topics, "
              f"alpha={alpha}, beta={beta} ===")
    else:
        num_topics, alpha, beta, out_dir = (
            ITER2_NUM_TOPICS, ITER2_ALPHA, ITER2_BETA, LDA_ONTOLOGY_DIR,
        )
        print(f"=== Iteration 2: ontology-replaced tokens, {num_topics} topics, "
              f"alpha={alpha}, beta={beta} ===")

    out_dir.mkdir(parents=True, exist_ok=True)

    df, texts = load_documents(args.mode)
    lda, vectorizer, feature_names, doc_topic = run_lda(texts, num_topics, alpha, beta)

    topics = top_words_per_topic(lda, feature_names)
    sizes = topic_sizes(topics, lda)

    print("\nTop terms per topic:")
    for topic_idx, words in topics.items():
        print(f"  Topic {topic_idx} (distinct-term size={sizes[topic_idx]}): "
              f"{', '.join(words)}")

    with open(out_dir / "topics.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in topics.items()}, f, indent=2)

    # --- topic proportions per issue ---
    dominant_topic = doc_topic.argmax(axis=1)
    dominant_share = doc_topic.max(axis=1)
    # number of topics an issue meaningfully touches on (share > 0.15)
    n_topics_per_issue = (doc_topic > 0.15).sum(axis=1)

    result = df[["issue_key"]].copy()
    result["dominant_topic"] = dominant_topic
    result["dominant_topic_share"] = dominant_share
    result["n_topics_present"] = n_topics_per_issue
    for i in range(num_topics):
        result[f"topic_{i}_proportion"] = doc_topic[:, i]

    result.to_csv(out_dir / "doc_topic_proportions.csv", index=False)

    print(f"\nIssues per dominant topic:\n{result['dominant_topic'].value_counts().sort_index()}")
    print(f"\nAverage number of topics touched per issue: {n_topics_per_issue.mean():.2f}")
    print(f"Median dominant-topic share (how 'peaked' the topics are): "
          f"{np.median(dominant_share):.2f}")

    print(f"\nSaved: {out_dir / 'topics.json'}")
    print(f"Saved: {out_dir / 'doc_topic_proportions.csv'}")


if __name__ == "__main__":
    main()
