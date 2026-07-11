import json
import time

import pandas as pd
import requests
from tqdm import tqdm

from bot_filter import filter_bot_comments, load_bot_authors
from config import (
    EXCEL_FILE,
    ISSUE_COLUMN,
    JIRA_API,
    OUTPUT_DIR,
    RAW_ISSUES_CSV,
    RAW_ISSUES_JSON,
)


def download_issues() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading issue IDs from {EXCEL_FILE}...")
    df = pd.read_excel(EXCEL_FILE)
    issue_keys = (
        df[ISSUE_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    print(f"Found {len(issue_keys)} issue IDs")

    bot_authors = load_bot_authors()
    print(f"Loaded {len(bot_authors)} bot author names to exclude")

    issues = []
    failed = []
    total_bot_comments_removed = 0

    print("Downloading issues from Jira...")
    for issue_key in tqdm(issue_keys):
        url = JIRA_API.format(issue_key)

        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                failed.append(issue_key)
                continue

            data = response.json()
            fields = data["fields"]

            parent = None
            if fields.get("parent"):
                parent = fields["parent"]["key"]

            components = []
            if fields.get("components"):
                components = [component["name"] for component in fields["components"]]

            assignee = None
            if fields.get("assignee"):
                assignee = fields["assignee"]["displayName"]

            reporter = None
            if fields.get("reporter"):
                reporter = fields["reporter"]["displayName"]

            description = fields.get("description") or ""

            raw_comments = (fields.get("comment") or {}).get("comments", [])
            human_comments = filter_bot_comments(raw_comments, bot_authors)
            bot_comments_removed = len(raw_comments) - len(human_comments)
            total_bot_comments_removed += bot_comments_removed

            issues.append(
                {
                    "issue_key": issue_key,
                    "summary": fields.get("summary"),
                    "description": description,
                    "issue_type": fields["issuetype"]["name"],
                    "status": fields["status"]["name"],
                    "priority": fields["priority"]["name"]
                    if fields.get("priority")
                    else None,
                    "resolution": fields["resolution"]["name"]
                    if fields.get("resolution")
                    else None,
                    "created": fields.get("created"),
                    "updated": fields.get("updated"),
                    "assignee": assignee,
                    "reporter": reporter,
                    "components": components,
                    "labels": fields.get("labels", []),
                    "comments": len(human_comments),
                    "bot_comments_removed": bot_comments_removed,
                    "parent": parent,
                }
            )

            time.sleep(0.2)
        except Exception as exc:
            print(f"Error downloading {issue_key}: {exc}")
            failed.append(issue_key)

    with open(RAW_ISSUES_JSON, "w", encoding="utf-8") as file:
        json.dump(issues, file, indent=4, ensure_ascii=False)

    pd.DataFrame(issues).to_csv(RAW_ISSUES_CSV, index=False, encoding="utf-8")

    print(f"Downloaded: {len(issues)} issues")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed issues:", ", ".join(failed))
    print(f"Total bot comments excluded from counts: {total_bot_comments_removed}")

    return issues


if __name__ == "__main__":
    download_issues()