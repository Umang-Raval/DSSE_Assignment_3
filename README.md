# DSSE Assignment 3 — Issue Topic Modeling Pipeline

This project has three parts, done in three weeks:

- **Week 1** — download Jira issues, clean/tokenize the text, build a
  vocabulary, and produce a document-term matrix.
- **Week 2** — run LDA topic modeling (baseline + ontology-based term
  replacement), tune the number of topics via coherence scores, and run
  BERTopic as a second algorithm. Answers RQ1.
- **Week 3** — statistically characterize each topic's issues (RQ2) and test
  co-occurrence between LDA/BERTopic topics and with design-decision types
  (RQ3).

> **Note:** the `vendor/mining-design-decisions` folder is a **separate,
> unrelated** repo that only supplies the compiled Rust text-cleaning
> accelerator used in Week 1. It is not part of this project's own code.

---

# WEEK 1 — Data Download & Preprocessing

## 1.1 What it does

1. **Determine the parent of an issue** (most issues have none) via the
   Jira API.
2. **Download issue data** (summary, description, type, status, priority,
   resolution, dates, assignee, reporter, components, labels, comment
   count, parent) and store it in JSON/CSV.
3. **Tokenize and pre-process** the concatenated summary + description:
   text cleaning (via the Rust accelerator), sentence/word tokenization,
   stop-word removal, POS tagging, and lemmatization.
4. **Build the vocabulary**: unique tokens across all issues with frequency
   counts, plus an analysis of the most frequent tokens and candidates for
   removal / ontology-mapping / further cleanup.
5. **Build the document-term matrix** — one row per issue, one column per
   vocabulary token — ready as input for LDA topic modeling.

Extra step added for data quality: comments authored by known **bot
accounts** (listed in `data/Bot_Comments.rtf`) are excluded from each
issue's comment count.

## 1.2 Folder structure

```
DSSE_Assignment_3/
├── data/
│   ├── Issues.xlsx            # issue IDs to download (column "Issue ID"),
│   │                          # AND (sheet "Yarn") the design-decision
│   │                          # labels used in Week 3
│   ├── Bot_Comments.rtf       # input: JSON list of bot author names to exclude
│   └── ontology_sheet_ref.xlsx # Week 2 input: reference ontology term list
├── output/
│   ├── raw_issues.json        # step 1-2 output
│   ├── raw_issues.csv
│   ├── processed_issues.json  # step 3 output: + tokens, clean_text
│   ├── processed_issues.csv
│   ├── vocabulary.csv         # step 4 output
│   ├── vocabulary_analysis.txt
│   ├── document_term_matrix.csv # step 5 output
│   └── topic_modelling/       # Week 2 + Week 3 output, see below
├── src/
│   ├── 01_download_jira.py
│   ├── 02_preprocess.py
│   ├── 03_build_vocabulary.py
│   ├── 04_document_term_matrix.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── bot_filter.py
│   ├── rust_accelerator.py
│   ├── text_cleaner.py
│   └── TopicModelling/        # Week 2 + Week 3 scripts, see below
├── vendor/
│   └── mining-design-decisions/deep_learning/dl_manager/
│       └── accelerator.cp3xx-win_amd64.pyd   # compiled Rust extension
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## 1.3 Tech stack

| Purpose                                | Tool / Library                                                                                 |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Jira data download                     | `requests`, `pandas`, `tqdm`                                                                   |
| Bot comment filtering                  | `striprtf`                                                                                     |
| Fast text cleaning / POS tagging       | Custom **Rust** extension (`dl_manager.accelerator`)                                           |
| Tokenization, stopwords, lemmatization | `nltk` (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger`) |
| Contraction expansion                  | `contractions`                                                                                 |
| Sentence/word tokenizing helper        | `gensim`                                                                                       |
| Excel reading                          | `pandas` + `openpyxl`                                                                          |

## 1.4 Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install setuptools_rust striprtf
```

### Build the Rust accelerator (Week 1 only — one-time)

Week 1's text cleaning depends on a compiled Rust extension. **This is only
needed for Week 1** — Weeks 2 and 3 do not use Rust at all.

Check whether `vendor/mining-design-decisions/deep_learning/dl_manager/`
already has a file like `accelerator.cp312-win_amd64.pyd` (Windows) or
`accelerator.cpython-312-*.so` (Linux/macOS). If it's there, skip straight
to running the pipeline. If not:

1. Install Rust: https://rustup.rs
2. From `vendor/mining-design-decisions/deep_learning/`:
   ```bash
   python setup.py build_ext --inplace
   ```
   On Windows this also needs the "Desktop development with C++" workload
   from Visual Studio Build Tools.

### Download NLTK data (usually automatic, can be run manually)

```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
```

## 1.5 Run

Full pipeline (downloads fresh from Jira, then preprocesses/vocab/DTM):

```bash
python run_pipeline.py
```

Reuse existing raw data, skip re-downloading from Jira:

```bash
python run_pipeline.py --skip-download
```

Run a single step manually (from inside `src/`, since scripts import
sibling modules directly):

```bash
cd src
python 01_download_jira.py
python 02_preprocess.py
python 03_build_vocabulary.py
python 04_document_term_matrix.py
```

## 1.6 Output files

- **`raw_issues.json` / `.csv`** — one record per issue with all Jira
  fields, bot-filtered comment count, and parent key (empty/`null` if none).
- **`processed_issues.json` / `.csv`** — same, plus `tokens` (cleaned,
  lemmatized token list) and `clean_text`.
- **`vocabulary.csv`** — every unique token with its frequency count.
- **`vocabulary_analysis.txt`** — top 30 tokens, plus candidates for
  removal / ontology-mapping / further preprocessing.
- **`document_term_matrix.csv`** — rows = issues, columns = vocabulary
  tokens, values = counts. Feeds into Week 2's LDA.

---

# WEEK 2 — Topic Modeling (LDA + BERTopic)

## 2.1 What it does

Two LDA iterations, ontology-based term replacement, coherence-based
topic-count tuning, and BERTopic as a second, independent algorithm — all
answering RQ1: what topics emerge from the issues, what are their common
keywords, and how many issues discuss each.

**Rust is not used anywhere in Week 2.** This week's stack is pure Python:
`gensim`, `scikit-learn`, `bertopic`, `sentence-transformers`, `umap-learn`,
`hdbscan`.

## 2.2 Folder structure

```
DSSE_Assignment_3/
├── data/
│   └── ontology_sheet_ref.xlsx        # reference term list (given)
├── output/
│   ├── processed_issues.csv           # from Week 1 (needs a 'tokens' column)
│   └── topic_modelling/               # everything below is generated
│       ├── ontology_term_map.json
│       ├── processed_issues_ontology.csv
│       ├── lda_baseline/
│       │   ├── topics.json
│       │   └── doc_topic_proportions.csv
│       ├── lda_ontology/
│       │   ├── topics.json
│       │   └── doc_topic_proportions.csv
│       ├── coherence_scores_baseline.csv / .png
│       ├── coherence_scores_ontology.csv / .png
│       └── bertopic/
│           ├── doc_topics.csv
│           ├── topic_info.csv
│           └── bertopic_model/
└── src/
    └── TopicModelling/
        ├── topic_config.py       # all paths + parameters
        ├── ontology_mapper.py    # Iteration 2 prep: ontology term replacement
        ├── lda_model.py          # Iteration 1 (baseline) + Iteration 2 (ontology) LDA
        ├── coherence_score.py    # topic-count tuning (3-10 topics)
        └── bertopic_model.py     # second algorithm, for RQ1
```

If `TopicModelling` sits at a different depth than `src/TopicModelling/`,
open `topic_config.py` and adjust the project-root line so it still points
at the folder containing `data/` and `output/`:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../TopicModelling -> up 2 levels
```

- `TopicModelling` directly under project root → `parents[1]`
- `TopicModelling` under `src/` (as shown above) → `parents[2]`

## 2.3 Install

```bash
pip install gensim scikit-learn matplotlib openpyxl bertopic sentence-transformers umap-learn hdbscan
```

`bertopic`/`sentence-transformers` download a small (~90MB) sentence-transformer
embedding model the first time you run it — that's expected, not an error.
No Rust build, no Rust toolchain needed for anything in this section.

## 2.4 Prerequisites (must already exist before running anything here)

- `output/processed_issues.csv` — from Week 1, must have a `tokens` column
- `data/ontology_sheet_ref.xlsx` — the reference ontology term sheet

## 2.5 Run order

From `src/TopicModelling/`:

```bash
# 1. Build the ontology term map + apply it to the tokens
python ontology_mapper.py

# 2. LDA Iteration 1 — baseline tokens
python lda_model.py baseline

# 3. LDA Iteration 2 — ontology-replaced tokens (needs step 1's output)
python lda_model.py ontology

# 4. Coherence sweep, 3-10 topics, baseline tokens
python coherence_score.py baseline

# 5. Coherence sweep, 3-10 topics, ontology tokens (needs step 1's output)
python coherence_score.py ontology

# --- STOP: check the coherence CSVs/plots, note the best topic count for
#     each, and update ITER1_NUM_TOPICS / ITER2_NUM_TOPICS in
#     topic_config.py to match. Then rerun steps 2 and 3 so your final
#     reported topics use the tuned count, not the placeholder default. ---

python lda_model.py baseline
python lda_model.py ontology

# 6. BERTopic — second algorithm required for RQ1 (independent of steps 1-5)
python bertopic_model.py
```

**Dependency notes:**

- Steps 3 and 5 require step 1 to have run first (they read
  `processed_issues_ontology.csv`).
- Steps 2, 4, and 6 only need `processed_issues.csv` and can run any time,
  but the order above matches the assignment's flow: baseline → judge it →
  ontology replacement → judge that → tune topic count → finalize.
- Step 6 (BERTopic) is independent of everything else — run it whenever.

## 2.6 What each script reports

| Script                                         | Assignment step                               | What to look at                                                                                                                   |
| ---------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `lda_model.py baseline`                        | Iteration 1                                   | Top terms per topic — useful (security, upgrades, refactoring, components) vs. project-specific noise                             |
| `ontology_mapper.py` + `lda_model.py ontology` | Iteration 2                                   | Did topics change after ontology replacement? Compare against Iteration 1's `topics.json`                                         |
| `coherence_score.py` (both modes)              | Optimize number of topics                     | Peak of the coherence curve = best topic count. Also check `lda_model.py`'s console output for topics with too few distinct terms |
| `doc_topic_proportions.csv` (either mode)      | Topic proportions per issue / dominant topics | `dominant_topic`, `dominant_topic_share`, `n_topics_present` columns                                                              |
| `bertopic_model.py`                            | Second algorithm for RQ1                      | `topic_info.csv` for keyword lists + issue counts per topic                                                                       |

RQ1 is answered by combining `lda_baseline/topics.json`,
`lda_ontology/topics.json`, and `bertopic/topic_info.csv` with the issue
counts already in `doc_topic_proportions.csv` / `doc_topics.csv` — label
each topic in plain language and report counts.

## 2.7 BERTopic pipeline — explicit 6-step configuration

`bertopic_model.py` builds BERTopic from six independently-configured,
named components (matching the reference pipeline from the course
material) instead of relying on BERTopic's un-configured defaults:

| Step                     | Component                                                                                                | What it does                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1. Extract embeddings    | `SentenceTransformer('all-MiniLM-L6-v2')`                                                                | Encodes each issue into a semantic vector                                            |
| 2. Reduce dimensionality | `UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')`                                    | Compresses embeddings for clustering                                                 |
| 3. Cluster               | `HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)` | Density-based clustering; unassigned issues go to topic `-1`                         |
| 4. Tokenize topics       | `CountVectorizer(stop_words='english')`                                                                  | Per-cluster bag-of-words                                                             |
| 5. Topic representation  | `ClassTfidfTransformer()`                                                                                | c-TF-IDF keyword scoring                                                             |
| 6. Fine-tune keywords    | `KeyBERTInspired()`                                                                                      | Re-ranks keywords by semantic relevance — cleaner top-word lists than c-TF-IDF alone |

All six are passed explicitly into `BERTopic(embedding_model=..., umap_model=...,
hdbscan_model=..., vectorizer_model=..., ctfidf_model=..., representation_model=...)`.
Parameters for steps 1-4 live in `topic_config.py`
(`BERTOPIC_EMBEDDING_MODEL`, `BERTOPIC_UMAP_PARAMS`, `BERTOPIC_HDBSCAN_PARAMS`,
`BERTOPIC_VECTORIZER_PARAMS`) — change them there, not in the script.

## 2.8 Known gotchas

- **Pipeline-artifact tokens** (`simpleclassname`, `classname`,
  `versionnumber`, `filepath`, `noformatblock`) aren't real vocabulary —
  they're redaction placeholders from Week 1's cleaning step, filtered via
  `EXTRA_STOPWORDS` in `topic_config.py` for **both** LDA and BERTopic. Add
  new ones there if they show up as top terms.
- **`ontology_sheet_ref.xlsx` is an example term sheet**, not a literal
  schema — its column names (`Connector_Data_`, `Pattern_`, etc.) get
  translated into the assignment's actual 5 classes (Component, Connector,
  Data, Solution, Quality Attribute) inside `ontology_mapper.py`.
- **Quality attributes are split per-attribute** (`QA_SECURITY`,
  `QA_PERFORMANCE`, etc.) via a curated dictionary in `topic_config.py`,
  since the sheet lists ~145 ungrouped terms. Terms with no curated bucket
  are left unreplaced rather than dumped in a catch-all.
- **`TECHNOLOGY` class is disabled by default** (`INCLUDE_TECHNOLOGY_CLASS
= False` in `topic_config.py`). An earlier run found it swallowed 1,142
  of 1,651 ontology terms (69%) — far more than every other class combined
  — and was a real contributor to Iteration 2's lower coherence score
  versus Iteration 1. Re-enable it only if you first prune that column
  down to a short, curated technology-name list.
- **Ontology replacement can still dilute topics if overused** even with
  `TECHNOLOGY` disabled — `CONNECTOR`/`DATA`/`SOLUTION` dominating most
  topics means `MIN_FREQ_TO_REPLACE` (currently 40) is too low. Raise the
  threshold or restrict replacement to `Quality_Attribute_`/`Pattern_`
  columns only.

---

# WEEK 3 — Topic Characteristics & Co-occurrence (RQ2 + RQ3)

## 3.1 What it does

1. **Build one merged per-issue table** joining topic assignments from all
   three models (LDA baseline, LDA ontology, BERTopic) with issue metadata
   (comments, description length, issue type) and design-decision labels.
2. **RQ2** — for each topic model, test whether comment count, description
   length, and issue type differ significantly by topic (box plots +
   Kruskal-Wallis + Dunn's post-hoc + chi-square).
3. **RQ3** — test significance of co-occurrence (a) between LDA and
   BERTopic topic assignments, and (b) between every topic model and each
   of the three design-decision types, via contingency tables, chi-square,
   Cramér's V, and standardized residuals.

**No Rust, no new heavy ML dependencies.** Pure `pandas` + `scipy` +
`scikit-posthocs` + `matplotlib`.

## 3.2 Folder structure

```
DSSE_Assignment_3/
├── data/
│   └── Issues.xlsx                     # sheet "Yarn": design-decision labels
├── output/
│   └── topic_modelling/
│       ├── ... (Week 2 outputs, unchanged) ...
│       ├── analysis_table.csv          # step 1 output — merged per-issue table
│       ├── rq2_characteristics/
│       │   ├── baseline/
│       │   ├── ontology/
│       │   └── bertopic/
│       │       ├── comments_boxplot.png
│       │       ├── description_length_boxplot.png
│       │       ├── issue_type_by_topic_contingency.csv
│       │       ├── dunn_posthoc_comments.csv          (if significant)
│       │       ├── dunn_posthoc_description_length.csv (if significant)
│       │       └── topic_summary_stats.csv
│       └── rq3_cooccurrence/
│           ├── lda_baseline_vs_bertopic/
│           ├── lda_ontology_vs_bertopic/
│           └── <topic_col>_vs_dd_<existence|property|executive>/  (9 folders)
│               ├── contingency_table.csv
│               ├── standardized_residuals.csv   (if significant)
│               └── significant_pairs.csv        (if significant)
└── src/
    └── TopicModelling/
        ├── topic_config.py            # extended with Week 3 paths/params
        ├── build_analysis_table.py    # step 1
        ├── rq2_characteristics.py     # step 2
        └── rq3_cooccurrence.py        # step 3
```

## 3.3 Install

```bash
pip install scikit-posthocs
```

(Everything else — `pandas`, `scipy`, `matplotlib` — is already installed
from Weeks 1-2.)

## 3.4 Prerequisites

- All of Week 2's outputs must already exist (`lda_baseline/`,
  `lda_ontology/`, `bertopic/` under `output/topic_modelling/`).
- `data/Issues.xlsx`, sheet `Yarn` — holds the design-decision labels.
  Column `Issue ID` must match `processed_issues.csv`'s `issue_key` exactly
  (verified: all 1,545 keys match with zero missing merges).

### About the design-decision labels

The `Types of design decisions` column in `Issues.xlsx` is a single string
of **three space-separated booleans**, e.g. `"True False False"` — an issue
can belong to more than one design-decision type at once (multi-label, not
one-of-three). `build_analysis_table.py` splits this into three separate
boolean columns (`dd_existence`, `dd_property`, `dd_executive` by default).

⚠️ **The column order (Existence / Property / Executive) is an assumption**,
based on the standard 3-way design-decision taxonomy from issue-mining
literature — the spreadsheet itself doesn't label which boolean is which.
Confirm this against your course material; if it's wrong, reorder
`DESIGN_DECISION_TYPE_NAMES` in `topic_config.py` and rerun
`build_analysis_table.py`.

## 3.5 Run order

From `src/TopicModelling/`:

```bash
# 1. Merge everything into one per-issue table
python build_analysis_table.py

# 2. RQ2 — run once per topic model
python rq2_characteristics.py baseline
python rq2_characteristics.py ontology
python rq2_characteristics.py bertopic

# 3. RQ3 — one run covers all comparisons (LDA-vs-BERTopic + all 9
#    topic-vs-design-decision tests)
python rq3_cooccurrence.py
```

## 3.6 What each script reports

| Script                           | Assignment step                                                  | What to look at                                                                                                                                                                      |
| -------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `build_analysis_table.py`        | data prep                                                        | Console confirms all 1,545 issues merged with no missing labels                                                                                                                      |
| `rq2_characteristics.py <model>` | RQ2: issue characteristics per topic                             | Kruskal-Wallis p-value (comments, description length) + Dunn's post-hoc pairs + chi-square (issue type) printed to console; box plots + CSVs saved to `rq2_characteristics/<model>/` |
| `rq3_cooccurrence.py`            | RQ3(a): LDA vs BERTopic; RQ3(b): topics vs design-decision types | Chi-square + Cramér's V per comparison; standardized residuals flag which specific topic/decision-type pairs co-occur significantly more/less than chance                            |

RQ2's answer combines the three `topic_summary_stats.csv` files with the
printed significance results. RQ3's answer combines the printed Cramér's V
values (overall association strength) with `significant_pairs.csv` (which
specific pairs drive that association) for each of the 11 comparisons run
(2 LDA-vs-BERTopic + 9 topic-vs-design-decision).

## 3.7 Known gotchas

- **BERTopic's outlier group (`-1`) is excluded** from every RQ2/RQ3 test
  involving BERTopic topics — it isn't a real topic, and including it would
  contaminate the statistics (42% of the corpus falls into it).
- **Design-decision columns are multi-label booleans, not one category** —
  `rq3_cooccurrence.py` runs three independent tests per topic model (one
  per decision type), not one combined categorical test. Don't try to
  collapse `dd_existence`/`dd_property`/`dd_executive` into a single column
  — an issue can be `True` on more than one.
- **Significant ≠ strong** — several RQ2/RQ3 tests are statistically
  significant (`p < 0.05`) but have weak Cramér's V (~0.1-0.15, e.g. issue
  type vs. LDA topic). Report both the p-value and the effect size; a tiny
  p-value on a large sample doesn't imply a practically important
  association.
- **LDA Iteration 2 (ontology tokens) is the model most likely to come back
  non-significant** on any given RQ2/RQ3 test (e.g. comments-vs-topic,
  Existence-vs-topic both came back non-significant in our run) — this is
  consistent with the Week 2 finding that ontology replacement diluted
  topic distinctiveness, not a bug in the Week 3 scripts.
