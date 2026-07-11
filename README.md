# DSSE Assignment 3 — Issue Topic Modeling Pipeline

This project has two parts, done in two weeks:

- **Week 1** — download Jira issues, clean/tokenize the text, build a
  vocabulary, and produce a document-term matrix.
- **Week 2** — run LDA topic modeling (baseline + ontology-based term
  replacement), tune the number of topics via coherence scores, and run
  BERTopic as a second algorithm.

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
│   ├── Issues.xlsx            # input: issue IDs to download (column "Issue ID")
│   └── Bot_Comments.rtf       # input: JSON list of bot author names to exclude
├── output/
│   ├── raw_issues.json        # step 1-2 output
│   ├── raw_issues.csv
│   ├── processed_issues.json  # step 3 output: + tokens, clean_text
│   ├── processed_issues.csv
│   ├── vocabulary.csv         # step 4 output
│   ├── vocabulary_analysis.txt
│   └── document_term_matrix.csv # step 5 output
├── src/
│   ├── 01_download_jira.py
│   ├── 02_preprocess.py
│   ├── 03_build_vocabulary.py
│   ├── 04_document_term_matrix.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── bot_filter.py
│   ├── rust_accelerator.py
│   └── text_cleaner.py
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
needed for Week 1** — Week 2 does not use Rust at all.

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
`gensim`, `scikit-learn`, `bertopic`, `sentence-transformers`.

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
pip install gensim scikit-learn matplotlib openpyxl bertopic sentence-transformers
```

`bertopic` downloads a small (~90MB) sentence-transformer embedding model
the first time you run it — that's expected, not an error. No Rust build,
no Rust toolchain needed for anything in this section.

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

## 2.7 Known gotchas

- **Pipeline-artifact tokens** (`simpleclassname`, `classname`,
  `versionnumber`, `filepath`, `noformatblock`) aren't real vocabulary —
  they're redaction placeholders from Week 1's cleaning step, filtered via
  `EXTRA_STOPWORDS` in `topic_config.py`. Add new ones there if they show
  up as top terms.
- **`ontology_sheet_ref.xlsx` is an example term sheet**, not a literal
  schema — its column names (`Connector_Data_`, `Pattern_`, etc.) get
  translated into the assignment's actual 5 classes (Component, Connector,
  Data, Solution, Quality Attribute) inside `ontology_mapper.py`.
- **Quality attributes are split per-attribute** (`QA_SECURITY`,
  `QA_PERFORMANCE`, etc.) via a curated dictionary in `topic_config.py`,
  since the sheet lists ~145 ungrouped terms. Terms with no curated bucket
  are left unreplaced rather than dumped in a catch-all.
- **Ontology replacement can dilute topics if overused** — `CONNECTOR`/
  `DATA`/`SOLUTION` dominating most topics means `MIN_FREQ_TO_REPLACE`
  (currently 40) is too low, or too many classes are merged in. Raise the
  threshold or restrict replacement to `Quality_Attribute_`/`Pattern_`
  columns only.
