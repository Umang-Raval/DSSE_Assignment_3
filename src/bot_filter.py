"""Utilities for excluding bot-authored comments from Jira issue data."""

import json

from striprtf.striprtf import rtf_to_text

from config import BOT_AUTHORS_FILE


def load_bot_authors() -> set[str]:
    """Load the list of bot display names and normalize them for matching."""
    raw = BOT_AUTHORS_FILE.read_text(encoding="utf-8", errors="ignore")

    # The file may be a genuine RTF document (starts with "{\rtf1...") or
    # plain text that just happens to have a .rtf extension. Handle both.
    if raw.lstrip().startswith("{\\rtf"):
        text = rtf_to_text(raw)
    else:
        text = raw

    # Extract just the JSON array in case there's any extra whitespace
    # or stray characters left over around it.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(
            f"Could not find a JSON array in {BOT_AUTHORS_FILE}. "
            "Open the file and confirm it contains a list like "
            '[\"Hive QA\", \"Hadoop QA\", ...]'
        )
    names = json.loads(text[start : end + 1])

    return {name.strip().lower() for name in names}


def is_bot_comment(comment: dict, bot_authors: set[str]) -> bool:
    author = comment.get("author") or {}
    display_name = (author.get("displayName") or "").strip().lower()
    return display_name in bot_authors


def filter_bot_comments(comments: list[dict], bot_authors: set[str]) -> list[dict]:
    """Return only the comments NOT authored by a bot."""
    return [c for c in comments if not is_bot_comment(c, bot_authors)]