#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import RAW_ISSUES_JSON  # noqa: E402
from importlib import import_module


def run_pipeline(skip_download: bool = False) -> None:
    if skip_download and RAW_ISSUES_JSON.exists():
        print(f"Skipping download; using existing {RAW_ISSUES_JSON}")
    else:
        if skip_download:
            print(
                f"No existing raw issues found at {RAW_ISSUES_JSON}; downloading anyway."
            )
        import_module("01_download_jira").download_issues()

    import_module("02_preprocess").run_preprocess()
    import_module("03_build_vocabulary").run_build_vocabulary()
    import_module("04_document_term_matrix").run_build_document_term_matrix()

    print("\nPipeline completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Jira issue preprocessing pipeline end-to-end."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse output/raw_issues.json instead of calling the Jira API.",
    )
    args = parser.parse_args()
    run_pipeline(skip_download=args.skip_download)


if __name__ == "__main__":
    main()
