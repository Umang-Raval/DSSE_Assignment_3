import json

import pandas as pd

from config import (
    OUTPUT_DIR,
    PROCESSED_ISSUES_CSV,
    PROCESSED_ISSUES_JSON,
    RAW_ISSUES_JSON,
)
from preprocessing import preprocess_issues


def run_preprocess() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_ISSUES_JSON, "r", encoding="utf-8") as file:
        raw_issues = json.load(file)

    print(f"Loaded {len(raw_issues)} raw issues")
    processed_issues = preprocess_issues(raw_issues)

    with open(PROCESSED_ISSUES_JSON, "w", encoding="utf-8") as file:
        json.dump(processed_issues, file, indent=4, ensure_ascii=False)

    pd.DataFrame(processed_issues).to_csv(
        PROCESSED_ISSUES_CSV,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved processed issues to {PROCESSED_ISSUES_JSON}")
    return processed_issues


if __name__ == "__main__":
    run_preprocess()
