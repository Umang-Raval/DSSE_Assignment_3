import json
from collections import Counter

import pandas as pd

from config import DOCUMENT_TERM_MATRIX_CSV, OUTPUT_DIR, PROCESSED_ISSUES_JSON


def run_build_document_term_matrix() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_ISSUES_JSON, "r", encoding="utf-8") as file:
        processed_issues = json.load(file)

    vocabulary = sorted(
        {
            token
            for issue in processed_issues
            for token in issue["tokens"]
        }
    )

    rows = []
    for issue in processed_issues:
        counts = Counter(issue["tokens"])
        row = {"issue_key": issue["issue_key"]}
        row.update({word: counts.get(word, 0) for word in vocabulary})
        rows.append(row)

    dtm = pd.DataFrame(rows)
    column_order = ["issue_key"] + vocabulary
    dtm = dtm[column_order]
    dtm.to_csv(DOCUMENT_TERM_MATRIX_CSV, index=False)

    print(f"Document-term matrix shape: {dtm.shape}")
    print(f"Saved DTM to {DOCUMENT_TERM_MATRIX_CSV}")
    return dtm


if __name__ == "__main__":
    run_build_document_term_matrix()
