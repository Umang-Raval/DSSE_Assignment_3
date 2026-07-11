"""
Week 3, RQ2: "What are the characteristics of the issues in each topic?"

For each topic model (LDA baseline, LDA ontology, BERTopic):
  - box plots of comments-per-issue and description-length-per-issue, by topic
  - Kruskal-Wallis test (non-parametric - comment/description counts are
    skewed, not normally distributed) to check if any topic differs on these
    continuous measures
  - if significant, Dunn's post-hoc test (Bonferroni-corrected) to find which
    specific topic pairs differ
  - Chi-square test of independence for issue_type vs topic

Usage:
    python rq2_characteristics.py baseline
    python rq2_characteristics.py ontology
    python rq2_characteristics.py bertopic
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

from topic_config import ANALYSIS_TABLE_CSV, RQ2_OUTPUT_DIR, SIGNIFICANCE_ALPHA

TOPIC_COLUMNS = {
    "baseline": "lda_baseline_topic",
    "ontology": "lda_ontology_topic",
    "bertopic": "bertopic_topic",
}


def box_plot(df: pd.DataFrame, topic_col: str, value_col: str, title: str, out_path):
    topics = sorted(df[topic_col].dropna().unique())
    data = [df.loc[df[topic_col] == t, value_col].dropna() for t in topics]

    fig, ax = plt.subplots(figsize=(max(8, len(topics) * 0.8), 5))
    ax.boxplot(data, tick_labels=[str(int(t)) for t in topics], showfliers=False)
    ax.set_xlabel("Topic")
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(title)
    plt.xticks(rotation=45 if len(topics) > 10 else 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def kruskal_and_posthoc(df: pd.DataFrame, topic_col: str, value_col: str, label: str):
    topics = sorted(df[topic_col].dropna().unique())
    groups = [df.loc[df[topic_col] == t, value_col].dropna() for t in topics]
    groups = [g for g in groups if len(g) > 0]

    stat, p = stats.kruskal(*groups)
    print(f"\n[{label}] Kruskal-Wallis on '{value_col}' across {len(groups)} topics: "
          f"H={stat:.3f}, p={p:.4g}")

    if p < SIGNIFICANCE_ALPHA:
        print(f"  -> significant (p < {SIGNIFICANCE_ALPHA}). Running Dunn's post-hoc "
              f"(Bonferroni-corrected)...")
        posthoc = sp.posthoc_dunn(
            df.dropna(subset=[topic_col, value_col]),
            val_col=value_col,
            group_col=topic_col,
            p_adjust="bonferroni",
        )
        sig_pairs = []
        for i, t1 in enumerate(posthoc.index):
            for t2 in posthoc.columns[i + 1:]:
                if posthoc.loc[t1, t2] < SIGNIFICANCE_ALPHA:
                    sig_pairs.append((t1, t2, posthoc.loc[t1, t2]))
        if sig_pairs:
            print(f"  Significantly different topic pairs ({value_col}):")
            for t1, t2, p_adj in sorted(sig_pairs, key=lambda x: x[2]):
                print(f"    topic {int(t1)} vs topic {int(t2)}: p_adj={p_adj:.4g}")
        else:
            print("  No individual pairs survived Bonferroni correction "
                  "(omnibus test significant, but pairwise differences are weak).")
        return posthoc
    else:
        print(f"  -> not significant (p >= {SIGNIFICANCE_ALPHA}). "
              f"Topics don't differ meaningfully on {value_col}.")
        return None


def chi_square_issue_type(df: pd.DataFrame, topic_col: str, label: str):
    contingency = pd.crosstab(df[topic_col], df["issue_type"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n[{label}] Chi-square test: issue_type vs topic: "
          f"chi2={chi2:.3f}, dof={dof}, p={p:.4g}")
    if p < SIGNIFICANCE_ALPHA:
        print(f"  -> significant (p < {SIGNIFICANCE_ALPHA}). Issue type distribution "
              f"depends on topic.")
    else:
        print(f"  -> not significant (p >= {SIGNIFICANCE_ALPHA}).")
    return contingency, chi2, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["baseline", "ontology", "bertopic"])
    args = parser.parse_args()
    topic_col = TOPIC_COLUMNS[args.model]

    out_dir = RQ2_OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_TABLE_CSV)
    if args.model == "bertopic":
        # outliers (-1) aren't a real topic - exclude from characteristic analysis
        n_before = len(df)
        df = df[df[topic_col] != -1]
        print(f"Excluded {n_before - len(df)} outlier issues (topic -1)")

    print(f"=== RQ2: {args.model} ({df[topic_col].nunique()} topics, "
          f"{len(df)} issues) ===")

    # --- box plots ---
    box_plot(df, topic_col, "comments", f"Comments per issue by topic ({args.model})",
              out_dir / "comments_boxplot.png")
    box_plot(df, topic_col, "description_length",
              f"Description length per issue by topic ({args.model})",
              out_dir / "description_length_boxplot.png")

    # --- significance tests ---
    kw_comments = kruskal_and_posthoc(df, topic_col, "comments", args.model)
    kw_desc = kruskal_and_posthoc(df, topic_col, "description_length", args.model)
    contingency, chi2, p = chi_square_issue_type(df, topic_col, args.model)

    contingency.to_csv(out_dir / "issue_type_by_topic_contingency.csv")
    if kw_comments is not None:
        kw_comments.to_csv(out_dir / "dunn_posthoc_comments.csv")
    if kw_desc is not None:
        kw_desc.to_csv(out_dir / "dunn_posthoc_description_length.csv")

    summary = df.groupby(topic_col).agg(
        n_issues=("issue_key", "count"),
        median_comments=("comments", "median"),
        median_description_length=("description_length", "median"),
    )
    summary.to_csv(out_dir / "topic_summary_stats.csv")
    print(f"\nPer-topic summary:\n{summary}")

    print(f"\nSaved plots and stats to: {out_dir}")


if __name__ == "__main__":
    main()
