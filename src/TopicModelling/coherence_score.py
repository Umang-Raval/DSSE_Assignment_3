"""
Step: Optimize the number of topics - calculate & plot coherence score for
topic counts 3..10 to find the best number of topics.

We use gensim's LdaMulticore + CoherenceModel with 'c_v' coherence, which
correlates better with human judgment of topic interpretability than 'u_mass'
(the metric used in the provided coherence.py example). u_mass only needs the
corpus itself, while c_v also looks at word co-occurrence in a sliding window,
giving a more reliable "are these topics sensible" signal - which is exactly
what the assignment is asking us to judge.

Usage:
    python coherence_score.py baseline
    python coherence_score.py ontology
"""
import argparse
import ast

import gensim
import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel

from topic_config import (
    EXTRA_STOPWORDS,
    MAX_DF,
    MIN_DF,
    PROCESSED_ISSUES_CSV,
    PROCESSED_ONTOLOGY_CSV,
    TM_OUTPUT_DIR as OUTPUT_DIR,
)

MIN_TOPICS = 3
MAX_TOPICS = 10  # inclusive


def load_token_lists(mode: str) -> list[list[str]]:
    if mode == "baseline":
        df = pd.read_csv(PROCESSED_ISSUES_CSV)
        tokens = df["tokens"].apply(ast.literal_eval)
        return tokens.apply(lambda t: [w for w in t if w not in EXTRA_STOPWORDS]).tolist()
    elif mode == "ontology":
        df = pd.read_csv(PROCESSED_ONTOLOGY_CSV)
        return df["tokens_ontology"].apply(ast.literal_eval).tolist()
    raise ValueError("mode must be 'baseline' or 'ontology'")


def filter_extremes(dictionary: corpora.Dictionary, texts: list[list[str]]):
    # translate MIN_DF/MAX_DF (used for sklearn CountVectorizer elsewhere)
    # into gensim's no_below / no_above so both pipelines stay consistent
    dictionary.filter_extremes(no_below=MIN_DF, no_above=MAX_DF)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "ontology"])
    args = parser.parse_args()

    print(f"Loading tokens ({args.mode})...")
    texts = load_token_lists(args.mode)

    dictionary = corpora.Dictionary(texts)
    dictionary, corpus = filter_extremes(dictionary, texts)

    results = []
    for num_topics in range(MIN_TOPICS, MAX_TOPICS + 1):
        print(f"  Fitting LDA with {num_topics} topics...")
        lda = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            chunksize=100,
            passes=10,
            per_word_topics=True,
        )
        coherence_model = CoherenceModel(
            model=lda, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        score = coherence_model.get_coherence()
        print(f"    c_v coherence = {score:.4f}")
        results.append({"num_topics": num_topics, "coherence_c_v": score})

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / f"coherence_scores_{args.mode}.csv"
    results_df.to_csv(out_csv, index=False)

    best_row = results_df.loc[results_df["coherence_c_v"].idxmax()]
    print(f"\nBest number of topics: {int(best_row['num_topics'])} "
          f"(c_v = {best_row['coherence_c_v']:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["num_topics"], results_df["coherence_c_v"], marker="o")
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score (c_v)")
    plt.title(f"LDA coherence vs. number of topics ({args.mode} tokens)")
    plt.grid(True, alpha=0.3)
    out_plot = OUTPUT_DIR / f"coherence_scores_{args.mode}.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_plot}")


if __name__ == "__main__":
    main()
