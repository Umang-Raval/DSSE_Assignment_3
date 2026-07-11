"""
Week 3, step 1: build a single per-issue table that RQ2 and RQ3 both read
from. Merges:
  - issue_type, comments count, description length (from processed_issues.csv)
  - dominant topic from LDA baseline, LDA ontology, and BERTopic
  - (optional) design-decision type, if DESIGN_DECISION_SOURCE_FILE is set
    in topic_config.py

Usage:
    python build_analysis_table.py
"""
import ast

import pandas as pd

from topic_config import (
    ANALYSIS_TABLE_CSV,
    BERTOPIC_DOC_TOPICS_CSV,
    DESIGN_DECISION_KEY_COLUMN,
    DESIGN_DECISION_RAW_COLUMN,
    DESIGN_DECISION_SHEET,
    DESIGN_DECISION_SOURCE_FILE,
    DESIGN_DECISION_TYPE_NAMES,
    LDA_BASELINE_PROPORTIONS_CSV,
    LDA_ONTOLOGY_PROPORTIONS_CSV,
    PROCESSED_ISSUES_CSV,
    TM_OUTPUT_DIR,
)


def main():
    TM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PROCESSED_ISSUES_CSV}...")
    issues = pd.read_csv(PROCESSED_ISSUES_CSV)
    issues["tokens"] = issues["tokens"].apply(ast.literal_eval)

    table = issues[["issue_key", "issue_type", "comments"]].copy()
    table["description_length"] = issues["tokens"].apply(len)  # word count, post-cleaning

    # --- LDA baseline dominant topic ---
    print(f"Loading {LDA_BASELINE_PROPORTIONS_CSV}...")
    base = pd.read_csv(LDA_BASELINE_PROPORTIONS_CSV)[["issue_key", "dominant_topic"]]
    base = base.rename(columns={"dominant_topic": "lda_baseline_topic"})
    table = table.merge(base, on="issue_key", how="left")

    # --- LDA ontology dominant topic ---
    print(f"Loading {LDA_ONTOLOGY_PROPORTIONS_CSV}...")
    onto = pd.read_csv(LDA_ONTOLOGY_PROPORTIONS_CSV)[["issue_key", "dominant_topic"]]
    onto = onto.rename(columns={"dominant_topic": "lda_ontology_topic"})
    table = table.merge(onto, on="issue_key", how="left")

    # --- BERTopic topic ---
    print(f"Loading {BERTOPIC_DOC_TOPICS_CSV}...")
    bert = pd.read_csv(BERTOPIC_DOC_TOPICS_CSV)
    table = table.merge(bert, on="issue_key", how="left")

    # --- design-decision types (multi-label: an issue can be more than one) ---
    if DESIGN_DECISION_SOURCE_FILE is not None:
        print(f"Loading design-decision labels from {DESIGN_DECISION_SOURCE_FILE} "
              f"(sheet '{DESIGN_DECISION_SHEET}')...")
        dd = pd.read_excel(DESIGN_DECISION_SOURCE_FILE, sheet_name=DESIGN_DECISION_SHEET)
        dd = dd[[DESIGN_DECISION_KEY_COLUMN, DESIGN_DECISION_RAW_COLUMN]].rename(
            columns={DESIGN_DECISION_KEY_COLUMN: "issue_key"}
        )

        # "True False False" -> 3 separate boolean columns, in the order
        # given by DESIGN_DECISION_TYPE_NAMES
        split_vals = dd[DESIGN_DECISION_RAW_COLUMN].str.split(expand=True)
        if split_vals.shape[1] != len(DESIGN_DECISION_TYPE_NAMES):
            raise ValueError(
                f"Expected {len(DESIGN_DECISION_TYPE_NAMES)} boolean values per row "
                f"in '{DESIGN_DECISION_RAW_COLUMN}', found {split_vals.shape[1]}. "
                f"Check DESIGN_DECISION_TYPE_NAMES in topic_config.py."
            )
        for i, name in enumerate(DESIGN_DECISION_TYPE_NAMES):
            dd[f"dd_{name}"] = split_vals[i].map({"True": True, "False": False})

        dd = dd.drop(columns=[DESIGN_DECISION_RAW_COLUMN])
        table = table.merge(dd, on="issue_key", how="left")

        n_missing = table[f"dd_{DESIGN_DECISION_TYPE_NAMES[0]}"].isna().sum()
        if n_missing:
            print(f"  [warn] {n_missing} issues have no design-decision label after merge")
        else:
            print(f"  Merged design-decision labels for all {len(table)} issues")
    else:
        print("[info] DESIGN_DECISION_SOURCE_FILE is not set in topic_config.py - "
              "skipping design-decision merge. RQ3's topic-vs-design-decision "
              "co-occurrence test will not be available until this is set.")
        for name in DESIGN_DECISION_TYPE_NAMES:
            table[f"dd_{name}"] = pd.NA

    table.to_csv(ANALYSIS_TABLE_CSV, index=False)
    print(f"\nSaved: {ANALYSIS_TABLE_CSV}  ({len(table)} issues)")
    print(table.head())


if __name__ == "__main__":
    main()
