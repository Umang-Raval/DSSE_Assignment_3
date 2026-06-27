import re
from collections import Counter

import nltk

from config import FORMATTING, PREPROCESS_THREADS
from rust_accelerator import RustAcceleratorNotBuiltError, get_accelerator
from text_cleaner import clean_issue_text

# Extra filler words common in Jira issue text but not in NLTK stopwords.
EXTRA_STOPWORDS = {
    "also",
    "would",
    "could",
    "should",
    "need",
    "needs",
    "needed",
    "new",
    "make",
    "made",
    "get",
    "got",
    "use",
    "used",
    "using",
    "one",
    "two",
    "may",
    "might",
    "must",
    "please",
    "see",
    "etc",
    "eg",
    "ie",
    "time",
    "jira",
    "org",
    "add",
}

BASE64_LIKE_PATTERN = re.compile(r"^[a-z0-9+/=]{16,}$", re.IGNORECASE)

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


def ensure_nltk_data() -> None:
    packages = (
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    )
    for package in packages:
        nltk.download(package, quiet=True)


def get_stopwords() -> set[str]:
    ensure_nltk_data()
    return set(nltk.corpus.stopwords.words("english")) | EXTRA_STOPWORDS


def clean_documents(documents: list[str]) -> list[list[str]]:
    """Rust text cleaning followed by sentence splitting (generator.py pattern)."""
    accelerator = get_accelerator()
    print("Using Rust accelerator for text cleaning.")
    cleaned = accelerator.bulk_clean_text_parallel(
        documents,
        FORMATTING,
        PREPROCESS_THREADS,
    )
    return [clean_issue_text(document) for document in cleaned]


def tokenize_documents(cleaned_documents: list[list[str]]) -> list[list[list[str]]]:
    ensure_nltk_data()
    return [
        [nltk.word_tokenize(sentence.lower()) for sentence in document]
        for document in cleaned_documents
    ]


def load_rust_tagger_data() -> tuple[dict, set[str], dict[str, str]]:
    ensure_nltk_data()
    loaded = nltk.load(
        "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle"
    )
    if isinstance(loaded, tuple):
        weights, tagdict, classes = loaded
        return weights, classes, tagdict

    return loaded.model.weights, loaded.classes, loaded.tagdict


def pos_tag_tokenized(
    tokenized_documents: list[list[list[str]]],
) -> list[list[list[tuple[str, str]]]]:
    accelerator = get_accelerator()
    weights, classes, tagdict = load_rust_tagger_data()
    tagger = accelerator.Tagger(weights, classes, tagdict)
    print("Using Rust accelerator for POS tagging.")
    return tagger.bulk_tag_parallel(tokenized_documents, PREPROCESS_THREADS)


def is_valid_token(token: str, stopwords: set[str]) -> bool:
    if len(token) < 2:
        return False
    if token.isdigit():
        return False
    if token in stopwords:
        return False
    if BASE64_LIKE_PATTERN.match(token):
        return False
    if len(token) > 30:
        return False
    return True


def lemmatize_tagged_documents(
    tagged_documents: list[list[list[tuple[str, str]]]],
    stopwords: set[str],
) -> list[list[str]]:
    ensure_nltk_data()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    processed = []

    for document in tagged_documents:
        words = []
        for sentence in document:
            for word, tag in sentence:
                if not is_valid_token(word, stopwords):
                    continue
                words.append(
                    lemmatizer.lemmatize(word, POS_CONVERSION.get(tag, "n"))
                )
        processed.append(words)

    return processed


def preprocess_issues(issues: list[dict]) -> list[dict]:
    try:
        get_accelerator()
    except RustAcceleratorNotBuiltError as exc:
        raise RuntimeError(str(exc)) from exc

    documents = [
        f"{issue.get('summary') or ''}\n{issue.get('description') or ''}"
        for issue in issues
    ]

    print(f"Cleaning {len(documents)} issue texts with Rust...")
    cleaned = clean_documents(documents)

    print("Tokenizing...")
    tokenized = tokenize_documents(cleaned)

    print("POS tagging with Rust...")
    tagged = pos_tag_tokenized(tokenized)

    print("Removing stopwords and lemmatizing...")
    stopwords = get_stopwords()
    token_lists = lemmatize_tagged_documents(tagged, stopwords)

    processed = []
    for issue, tokens in zip(issues, token_lists):
        processed.append(
            {
                **issue,
                "tokens": tokens,
                "clean_text": " ".join(tokens),
            }
        )

    return processed


def build_vocabulary(processed_issues: list[dict]) -> Counter:
    counter = Counter()
    for issue in processed_issues:
        counter.update(issue["tokens"])
    return counter


def identify_vocabulary_candidates(counter: Counter) -> dict[str, list[tuple[str, int]]]:
    stopwords = get_stopwords()
    top_tokens = counter.most_common(30)

    remove_candidates = [
        (word, freq)
        for word, freq in counter.items()
        if word in stopwords or BASE64_LIKE_PATTERN.match(word)
    ]

    ontology_candidates = [
        (word, freq)
        for word, freq in counter.most_common(200)
        if word
        in {
            "yarn",
            "container",
            "application",
            "node",
            "resource",
            "queue",
            "scheduler",
            "cluster",
            "hadoop",
            "nodemanager",
            "resourcemanager",
            "capacity",
            "fair",
            "hdfs",
            "mapreduce",
            "docker",
            "gpu",
            "kubernetes",
        }
    ]

    noisy_candidates = sorted(
        [
            (word, freq)
            for word, freq in counter.items()
            if len(word) > 20 or (len(word) > 12 and word.isalpha())
        ],
        key=lambda item: (-item[1], item[0]),
    )[:30]

    return {
        "top_tokens": top_tokens,
        "remove_candidates": sorted(remove_candidates, key=lambda item: -item[1])[:30],
        "ontology_candidates": ontology_candidates,
        "noisy_candidates": noisy_candidates,
    }
