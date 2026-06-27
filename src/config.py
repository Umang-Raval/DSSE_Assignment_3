from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DL_MANAGER_DIR = (
    PROJECT_ROOT / "vendor" / "mining-design-decisions" / "deep_learning"
)

EXCEL_FILE = DATA_DIR / "Issues.xlsx"
RAW_ISSUES_JSON = OUTPUT_DIR / "raw_issues.json"
RAW_ISSUES_CSV = OUTPUT_DIR / "raw_issues.csv"
PROCESSED_ISSUES_JSON = OUTPUT_DIR / "processed_issues.json"
PROCESSED_ISSUES_CSV = OUTPUT_DIR / "processed_issues.csv"
VOCABULARY_CSV = OUTPUT_DIR / "vocabulary.csv"
VOCABULARY_ANALYSIS_TXT = OUTPUT_DIR / "vocabulary_analysis.txt"
DOCUMENT_TERM_MATRIX_CSV = OUTPUT_DIR / "document_term_matrix.csv"

JIRA_API = "https://issues.apache.org/jira/rest/api/2/issue/{}"
ISSUE_COLUMN = "Issue ID"

PREPROCESS_THREADS = 8
FORMATTING = "markers"
