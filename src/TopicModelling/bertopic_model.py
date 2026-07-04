"""
Step: Run BERTopic on issue summary+description to determine topics
(the second algorithm required for research question 1).

Unlike LDA, BERTopic works on raw-ish sentences (it embeds them with a
sentence-transformer), so we feed it the lightly-cleaned text rather than the
heavily stemmed/lemmatized token list - stemming actually hurts BERTopic
because it relies on semantic embeddings of natural language, not word
co-occurrence counts.

Install first:
    pip install bertopic sentence-transformers

Usage:
    python bertopic_model.py
"""
import ast

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from topic_config import BERTOPIC_DIR, EXTRA_STOPWORDS, PROCESSED_ISSUES_CSV


def main():
    BERTOPIC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PROCESSED_ISSUES_CSV}...")
    df = pd.read_csv(PROCESSED_ISSUES_CSV)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)
    df["tokens"] = df["tokens"].apply(
        lambda toks: [t for t in toks if t not in EXTRA_STOPWORDS]
    )
    # BERTopic wants readable text, not a bag-of-stems -> re-join tokens with
    # spaces (already cleaned/lowercased from week 1) rather than raw summary
    # + description, so results are comparable to the LDA vocabulary.
    docs = df["tokens"].apply(lambda t: " ".join(t)).tolist()

    # Use the same min/max df filtering as LDA so topics aren't dominated by
    # simpleclassname / boilerplate tokens.
    vectorizer_model = CountVectorizer(min_df=5, max_df=0.6, stop_words="english")

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=False,
        verbose=True,
    )

    print("Fitting BERTopic (this downloads a sentence-transformer model on first run)...")
    topics, _ = topic_model.fit_transform(docs)

    df["bertopic_topic"] = topics
    df[["issue_key", "bertopic_topic"]].to_csv(
        BERTOPIC_DIR / "doc_topics.csv", index=False
    )

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(BERTOPIC_DIR / "topic_info.csv", index=False)

    print("\nTopic overview (topic -1 = outliers, not assigned to any topic):")
    print(topic_info.head(20))

    topic_model.save(str(BERTOPIC_DIR / "bertopic_model"), serialization="safetensors")
    print(f"\nSaved: {BERTOPIC_DIR / 'doc_topics.csv'}")
    print(f"Saved: {BERTOPIC_DIR / 'topic_info.csv'}")
    print(f"Saved model to: {BERTOPIC_DIR / 'bertopic_model'}")


if __name__ == "__main__":
    main()
