import json
import pickle
from pathlib import Path

import nltk

from dl_manager import accelerator

# -----------------------------
# CONFIG
# -----------------------------

INPUT_FOLDER = Path("data/issues")
OUTPUT_FOLDER = Path("data/processed")
OUTPUT_FOLDER.mkdir(exist_ok=True)

THREADS = 8
FORMATTING = "markers"

USE_LOWERCASE = True
REMOVE_STOPWORDS = True
USE_LEMMATIZATION = True

# -----------------------------
# NLTK
# -----------------------------

stopwords = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()

POS_CONVERSION = {
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
}

# -----------------------------
# Rust POS Tagger
# -----------------------------

weights, tagdict, classes = nltk.load(
    "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle"
)

tagger = accelerator.Tagger(
    weights,
    classes,
    tagdict
)

# -----------------------------
# Read Issues
# -----------------------------

documents = []
issue_ids = []

for file in sorted(INPUT_FOLDER.glob("*.json")):

    with open(file, encoding="utf8") as f:
        issue = json.load(f)

    summary = issue["fields"].get("summary", "")
    description = issue["fields"].get("description", "")

    documents.append(summary + "\n" + description)
    issue_ids.append(issue["key"])

print(f"Loaded {len(documents)} issues")

# -----------------------------
# Rust Cleaning
# -----------------------------

cleaned = accelerator.bulk_clean_text_parallel(
    documents,
    FORMATTING,
    THREADS
)

# -----------------------------
# Sentence tokenize
# -----------------------------

tokenized = []

for doc in cleaned:

    sentences = nltk.sent_tokenize(doc)

    tokenized.append([
        nltk.word_tokenize(
            sentence.lower() if USE_LOWERCASE else sentence
        )
        for sentence in sentences
    ])

print("Tokenized.")

# -----------------------------
# Rust POS Tagging
# -----------------------------

tagged = tagger.bulk_tag_parallel(
    tokenized,
    THREADS
)

print("POS tagging done.")

# -----------------------------
# Lemmatization
# -----------------------------

processed = []

for issue in tagged:

    words = []

    for sentence in issue:

        for word, tag in sentence:

            if REMOVE_STOPWORDS and word in stopwords:
                continue

            if USE_LEMMATIZATION:

                word = lemmatizer.lemmatize(
                    word,
                    POS_CONVERSION.get(tag, "n")
                )

            words.append(word)

    processed.append(words)

print("Lemmatization complete.")

# -----------------------------
# Save
# -----------------------------

with open(
    OUTPUT_FOLDER / "processed.pkl",
    "wb"
) as f:

    pickle.dump(
        {
            "ids": issue_ids,
            "documents": processed
        },
        f
    )

print()
print("Saved to:")
print(OUTPUT_FOLDER / "processed.pkl")