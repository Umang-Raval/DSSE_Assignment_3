import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os

os.makedirs("output", exist_ok=True)

with open("output/processed_issues.json", "r", encoding="utf-8") as f:
    issues = json.load(f)

documents = [
    issue["clean_text"]
    for issue in issues
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

dtm = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

dtm.to_csv(
    "output/document_term_matrix.csv",
    index=False
)

print(dtm.shape)