import pandas as pd
import requests
import json
import time
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================

EXCEL_FILE = "data/Issues.xlsx"

OUTPUT_JSON = "output/raw_issues.json"
OUTPUT_CSV = "output/raw_issues.csv"

JIRA_API = "https://issues.apache.org/jira/rest/api/2/issue/{}"

# =====================================================
# READ ISSUE IDS
# =====================================================

print("Reading Excel file...")

df = pd.read_excel(EXCEL_FILE)

print("\nColumns in Excel:")
print(df.columns.tolist())

# -----------------------------------------------------
# CHANGE THIS IF YOUR COLUMN NAME IS DIFFERENT
# -----------------------------------------------------

ISSUE_COLUMN = "Issue ID"

issue_keys = (
    df[ISSUE_COLUMN]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

print(f"\nFound {len(issue_keys)} issue IDs")

# =====================================================
# DOWNLOAD ISSUES
# =====================================================

issues = []

failed = []

print("\nDownloading issues...\n")

for issue in tqdm(issue_keys):

    url = JIRA_API.format(issue)

    try:

        response = requests.get(url, timeout=30)

        if response.status_code != 200:

            failed.append(issue)
            continue

        data = response.json()

        fields = data["fields"]

        # ------------------------------------------
        # Parent Issue
        # ------------------------------------------

        parent = None

        if fields.get("parent"):

            parent = fields["parent"]["key"]

        # ------------------------------------------
        # Components
        # ------------------------------------------

        components = []

        if fields.get("components"):

            components = [
                c["name"]
                for c in fields["components"]
            ]

        # ------------------------------------------
        # Labels
        # ------------------------------------------

        labels = fields.get("labels", [])

        # ------------------------------------------
        # Assignee
        # ------------------------------------------

        assignee = None

        if fields.get("assignee"):

            assignee = fields["assignee"]["displayName"]

        # ------------------------------------------
        # Reporter
        # ------------------------------------------

        reporter = None

        if fields.get("reporter"):

            reporter = fields["reporter"]["displayName"]

        # ------------------------------------------
        # Description
        # ------------------------------------------

        description = fields.get("description")

        if description is None:

            description = ""

        # ------------------------------------------
        # Store Everything
        # ------------------------------------------

        issues.append({

            "issue_key": issue,

            "summary": fields.get("summary"),

            "description": description,

            "issue_type": fields["issuetype"]["name"],

            "status": fields["status"]["name"],

            "priority":
                fields["priority"]["name"]
                if fields.get("priority")
                else None,

            "resolution":
                fields["resolution"]["name"]
                if fields.get("resolution")
                else None,

            "created": fields.get("created"),

            "updated": fields.get("updated"),

            "assignee": assignee,

            "reporter": reporter,

            "components": components,

            "labels": labels,

            "comments":
                fields["comment"]["total"],

            "parent": parent

        })

        # Avoid hammering the API

        time.sleep(0.2)

    except Exception as e:

        print(f"Error downloading {issue}")

        print(e)

        failed.append(issue)

# =====================================================
# SAVE JSON
# =====================================================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

    json.dump(
        issues,
        f,
        indent=4,
        ensure_ascii=False
    )

# =====================================================
# SAVE CSV
# =====================================================

csv_df = pd.DataFrame(issues)

csv_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)

print("\nDone!")

print(f"Downloaded: {len(issues)} issues")

print(f"Failed: {len(failed)}")

if failed:

    print("\nFailed Issues:")

    for i in failed:

        print(i)