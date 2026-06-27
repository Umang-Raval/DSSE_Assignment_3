import json
import pandas as pd
from collections import Counter
import os

os.makedirs("output", exist_ok=True)

with open("output/processed_issues.json", "r", encoding="utf-8") as f:
    issues = json.load(f)

counter = Counter()

for issue in issues:
    counter.update(issue["tokens"])

vocab = pd.DataFrame(
    counter.items(),
    columns=["Word", "Frequency"]
)

vocab = vocab.sort_values(
    by="Frequency",
    ascending=False
)

vocab.to_csv(
    "output/vocabulary.csv",
    index=False
)

print(vocab.head(30))