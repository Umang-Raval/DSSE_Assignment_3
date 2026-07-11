"""
Week 3, RQ3: "What topics significantly co-occur together among LDA topics
and BERTopics? What topics from LDA and BERTopics significantly co-occur
with the types of design decisions?"

For each pair of topic assignments (LDA baseline vs BERTopic, LDA ontology
vs BERTopic, and - if available - each topic model vs design-decision type):
  - build a contingency table
  - Chi-square test of independence for the overall association
  - standardized (Pearson) residuals to find which specific cell
    (topic_A, topic_B) pairs co-occur significantly more/less than chance
  - Cramer's V as an effect-size summary of how strong the association is

Usage:
    python rq3_cooccurrence.py
"""
import numpy as np
import pandas as pd
from scipy import stats

from topic_config import (
    ANALYSIS_TABLE_CSV,
    DESIGN_DECISION_TYPE_NAMES,
    RQ3_OUTPUT_DIR,
    SIGNIFICANCE_ALPHA,
)

RESIDUAL_THRESHOLD = 2.0  # |standardized residual| > ~2 is a rough "significant cell" cutoff


def cramers_v(chi2: float, contingency: pd.DataFrame) -> float:
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1))))


def standardized_residuals(contingency: pd.DataFrame, expected: np.ndarray) -> pd.DataFrame:
    observed = contingency.to_numpy()
    n = observed.sum()
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    # standardized (adjusted) residuals - accounts for row/col marginals,
    # so they're comparable to a z-score (|value| > ~2 is noteworthy)
    denom = np.sqrt(expected * (1 - row_totals / n) * (1 - col_totals / n))
    resid = (observed - expected) / denom
    return pd.DataFrame(resid, index=contingency.index, columns=contingency.columns)


def analyze_cooccurrence(df: pd.DataFrame, col_a: str, col_b: str, label: str, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df.dropna(subset=[col_a, col_b])

    contingency = pd.crosstab(sub[col_a], sub[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    v = cramers_v(chi2, contingency)

    print(f"\n=== {label}: {col_a} x {col_b} ===")
    print(f"chi2={chi2:.2f}, dof={dof}, p={p:.4g}, Cramer's V={v:.3f} "
          f"({'strong' if v > 0.3 else 'moderate' if v > 0.1 else 'weak'} association)")

    contingency.to_csv(out_dir / "contingency_table.csv")

    if p >= SIGNIFICANCE_ALPHA:
        print(f"  -> not significant (p >= {SIGNIFICANCE_ALPHA}). "
              f"No overall evidence of association; skipping residual breakdown.")
        return

    resid = standardized_residuals(contingency, expected)
    resid.to_csv(out_dir / "standardized_residuals.csv")

    sig_cells = []
    for a in resid.index:
        for b in resid.columns:
            r = resid.loc[a, b]
            if abs(r) > RESIDUAL_THRESHOLD:
                sig_cells.append((a, b, r))
    sig_cells.sort(key=lambda x: -abs(x[2]))

    print(f"  Significant co-occurring pairs (|residual| > {RESIDUAL_THRESHOLD}):")
    for a, b, r in sig_cells[:25]:
        direction = "MORE than expected" if r > 0 else "LESS than expected"
        print(f"    {col_a}={a} & {col_b}={b}: residual={r:+.2f} ({direction})")
    if len(sig_cells) > 25:
        print(f"    ... and {len(sig_cells) - 25} more (see standardized_residuals.csv)")

    pd.DataFrame(sig_cells, columns=[col_a, col_b, "std_residual"]).to_csv(
        out_dir / "significant_pairs.csv", index=False
    )


def main():
    df = pd.read_csv(ANALYSIS_TABLE_CSV)

    # BERTopic outliers (-1) aren't a real topic - exclude from co-occurrence tests
    bert_valid = df["bertopic_topic"] != -1

    analyze_cooccurrence(
        df[bert_valid], "lda_baseline_topic", "bertopic_topic",
        "LDA baseline vs BERTopic", RQ3_OUTPUT_DIR / "lda_baseline_vs_bertopic",
    )
    analyze_cooccurrence(
        df[bert_valid], "lda_ontology_topic", "bertopic_topic",
        "LDA ontology vs BERTopic", RQ3_OUTPUT_DIR / "lda_ontology_vs_bertopic",
    )

    dd_columns = [f"dd_{name}" for name in DESIGN_DECISION_TYPE_NAMES]
    have_labels = all(col in df.columns for col in dd_columns) and df[dd_columns[0]].notna().any()

    if have_labels:
        topic_models = {
            "lda_baseline_topic": "LDA baseline",
            "lda_ontology_topic": "LDA ontology",
            "bertopic_topic": "BERTopic",
        }
        for topic_col, topic_label in topic_models.items():
            subset = df[bert_valid] if topic_col == "bertopic_topic" else df
            for dd_name, dd_col in zip(DESIGN_DECISION_TYPE_NAMES, dd_columns):
                analyze_cooccurrence(
                    subset, topic_col, dd_col,
                    f"{topic_label} vs design-decision:{dd_name}",
                    RQ3_OUTPUT_DIR / f"{topic_col}_vs_dd_{dd_name}",
                )
    else:
        print("\n[info] No design-decision labels in analysis_table.csv - "
              "set DESIGN_DECISION_SOURCE_FILE in topic_config.py and rerun "
              "build_analysis_table.py to enable the topic-vs-design-decision "
              "tests required by RQ3.")


if __name__ == "__main__":
    main()
