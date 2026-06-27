import json

import pandas as pd

from config import (
    OUTPUT_DIR,
    PROCESSED_ISSUES_JSON,
    VOCABULARY_ANALYSIS_TXT,
    VOCABULARY_CSV,
)
from preprocessing import build_vocabulary, identify_vocabulary_candidates


def write_vocabulary_analysis(candidates: dict[str, list[tuple[str, int]]]) -> None:
    lines = [
        "Vocabulary Analysis",
        "===================",
        "",
        "Top 30 tokens by frequency:",
    ]

    for word, freq in candidates["top_tokens"]:
        lines.append(f"  {word}: {freq}")

    lines.extend(
        [
            "",
            "Candidates for removal (stopwords / noisy tokens):",
        ]
    )
    for word, freq in candidates["remove_candidates"]:
        lines.append(f"  {word}: {freq}")

    lines.extend(
        [
            "",
            "Candidates for ontology classes (domain concepts):",
        ]
    )
    for word, freq in candidates["ontology_candidates"]:
        lines.append(f"  {word}: {freq}")

    lines.extend(
        [
            "",
            "Candidates for further preprocessing (long/noisy tokens):",
        ]
    )
    for word, freq in candidates["noisy_candidates"]:
        lines.append(f"  {word}: {freq}")

    VOCABULARY_ANALYSIS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_build_vocabulary() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_ISSUES_JSON, "r", encoding="utf-8") as file:
        processed_issues = json.load(file)

    counter = build_vocabulary(processed_issues)
    vocab = pd.DataFrame(counter.items(), columns=["Word", "Frequency"])
    vocab = vocab.sort_values(by="Frequency", ascending=False)
    vocab.to_csv(VOCABULARY_CSV, index=False)

    candidates = identify_vocabulary_candidates(counter)
    write_vocabulary_analysis(candidates)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Saved vocabulary to {VOCABULARY_CSV}")
    print(f"Saved analysis to {VOCABULARY_ANALYSIS_TXT}")
    print("\nTop 20 tokens:")
    print(vocab.head(20).to_string(index=False))

    return vocab


if __name__ == "__main__":
    run_build_vocabulary()
