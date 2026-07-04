"""
Step: Iteration 2 preparation - replace high-frequency vocabulary terms with
their ontology class (Component, Connector, Data, Pattern, Quality Attribute,
Technology) using the reference sheet from the LDA-on-blogs repo.

Only terms that:
  1. appear in one of the ontology columns we trust, AND
  2. occur often enough in our corpus (MIN_FREQ_TO_REPLACE)
are replaced. This avoids polluting rare/ambiguous tokens and follows the
assignment's guidance to "only replace terms that you are sure about" and to
prioritize terms with the highest frequency.
"""
import ast
import json
from collections import Counter

import pandas as pd

from topic_config import (
    EXTRA_STOPWORDS,
    MIN_FREQ_TO_REPLACE,
    ONTOLOGY_COLUMN_TO_CLASS,
    ONTOLOGY_MAP_JSON,
    ONTOLOGY_XLSX,
    PROCESSED_ISSUES_CSV,
    PROCESSED_ONTOLOGY_CSV,
    QUALITY_ATTRIBUTE_COLUMN,
    QUALITY_ATTRIBUTE_GROUPS,
    SPLIT_QUALITY_ATTRIBUTES,
    TM_OUTPUT_DIR as OUTPUT_DIR,
)


def build_ontology_dictionary() -> dict[str, str]:
    """Build {term: ONTOLOGY_CLASS} using the assignment's own class
    definitions (Component, Connector, Data, Solution, Quality attribute),
    translating the reference sheet's column names into those classes rather
    than using the sheet's raw headers verbatim."""
    df = pd.read_excel(ONTOLOGY_XLSX)

    term_to_class = {}

    # --- Component / Connector / Data / Solution / (optional) Technology ---
    for col, class_name in ONTOLOGY_COLUMN_TO_CLASS.items():
        if col not in df.columns:
            print(f"  [warn] column '{col}' not found in ontology sheet, skipping")
            continue
        terms = df[col].dropna().astype(str)
        for raw_term in terms:
            term = raw_term.strip().lower()
            if not term:
                continue
            # a term already claimed by another class is left to the first
            # class that claimed it, to avoid ambiguous double-mapping
            term_to_class.setdefault(term, class_name)

    # --- Quality attributes: single class, or split per attribute ---
    qa_terms = set(
        df[QUALITY_ATTRIBUTE_COLUMN].dropna().astype(str).str.strip().str.lower()
    )
    if SPLIT_QUALITY_ATTRIBUTES:
        curated = set()
        for qa_class, words in QUALITY_ATTRIBUTE_GROUPS.items():
            for term in words:
                term_to_class.setdefault(term, qa_class)
                curated.add(term)
        uncurated = qa_terms - curated
        if uncurated:
            print(f"  [info] {len(uncurated)} quality-attribute terms have no "
                  f"curated bucket and are left unreplaced: {sorted(uncurated)}")
    else:
        for term in qa_terms:
            term_to_class.setdefault(term, "QUALITY_ATTRIBUTE")

    n_classes = len(set(term_to_class.values()))
    print(f"Built ontology dictionary with {len(term_to_class)} terms "
          f"across {n_classes} classes")
    return term_to_class


def get_corpus_frequencies(token_lists: list[list[str]]) -> Counter:
    freq = Counter()
    for tokens in token_lists:
        freq.update(tokens)
    return freq


def apply_ontology_replacement(
    token_lists: list[list[str]],
    term_to_class: dict[str, str],
    corpus_freq: Counter,
    min_freq: int,
) -> tuple[list[list[str]], Counter]:
    """Replace tokens with their ontology class, only for high-frequency terms."""
    replaced_counter = Counter()
    new_token_lists = []

    for tokens in token_lists:
        new_tokens = []
        for tok in tokens:
            ontology_class = term_to_class.get(tok)
            if ontology_class is not None and corpus_freq[tok] >= min_freq:
                new_tokens.append(ontology_class)
                replaced_counter[tok] += 1
            else:
                new_tokens.append(tok)
        new_token_lists.append(new_tokens)

    return new_token_lists, replaced_counter


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading pre-processed issues from {PROCESSED_ISSUES_CSV}...")
    df = pd.read_csv(PROCESSED_ISSUES_CSV)
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

    print(f"Dropping pipeline-artifact stopwords: {sorted(EXTRA_STOPWORDS)}")
    df["tokens"] = df["tokens"].apply(
        lambda toks: [t for t in toks if t not in EXTRA_STOPWORDS]
    )

    print("Building ontology dictionary...")
    term_to_class = build_ontology_dictionary()
    with open(ONTOLOGY_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(term_to_class, f, indent=2)

    print("Computing corpus term frequencies...")
    corpus_freq = get_corpus_frequencies(df["tokens"].tolist())

    print(f"Applying replacement (min corpus frequency = {MIN_FREQ_TO_REPLACE})...")
    new_tokens, replaced_counter = apply_ontology_replacement(
        df["tokens"].tolist(), term_to_class, corpus_freq, MIN_FREQ_TO_REPLACE
    )
    df["tokens_ontology"] = new_tokens
    df["clean_text_ontology"] = df["tokens_ontology"].apply(lambda t: " ".join(t))

    df.to_csv(PROCESSED_ONTOLOGY_CSV, index=False, encoding="utf-8")

    print(f"\nReplaced {sum(replaced_counter.values())} token occurrences "
          f"across {len(replaced_counter)} unique terms.")
    print("\nTop 20 replaced terms (term -> class, count):")
    for term, count in replaced_counter.most_common(20):
        print(f"  {term:20s} -> {term_to_class[term]:12s}  ({count} occurrences)")

    print(f"\nSaved: {PROCESSED_ONTOLOGY_CSV}")
    print(f"Saved: {ONTOLOGY_MAP_JSON}")


if __name__ == "__main__":
    main()