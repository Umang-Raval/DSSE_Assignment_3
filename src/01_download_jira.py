import json
import time

import pandas as pd
import requests
from tqdm import tqdm

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

    issues = []
    failed = []

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
                    "comments": fields["comment"]["total"],
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

    return issues


if __name__ == "__main__":
    download_issues()
