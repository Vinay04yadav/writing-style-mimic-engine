"""Prepare raw books as separate, auditable text files for downstream work.

Paths are resolved from the project root, independently of the process working
directory. From the repository root, run ``python src/preprocess_corpus.py``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_books"
CATALOG_PATH = PROJECT_ROOT / "data" / "books.json"
BOUNDARIES_PATH = PROJECT_ROOT / "data" / "text_boundaries.json"

START_MARKER = re.compile(
    r"\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK\b[^\r\n]*",
    re.IGNORECASE,
)
END_MARKER = re.compile(
    r"\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK\b[^\r\n]*"
    r"|^\s*End of (?:the )?Project Gutenberg(?:'s)?\b[^\r\n]*",
    re.IGNORECASE | re.MULTILINE,
)


def clean_gutenberg_wrapper(text: str) -> tuple[str, str]:
    """Strip Gutenberg wrapper text when its boundary markers are available.

    Files without markers are retained as-is and explicitly marked for review;
    the cleaner does not guess where a book begins or silently discard its
    front matter.
    """
    starts = list(START_MARKER.finditer(text))
    ends = list(END_MARKER.finditer(text))

    start = starts[-1].end() if starts else 0
    end = ends[0].start() if ends else len(text)

    if starts and ends and end < start:
        return normalize_newlines(text), "markers_invalid_review"

    cleaned = text[start:end].strip()
    if starts and ends:
        status = "gutenberg_markers_removed"
    elif starts:
        status = "start_marker_removed_end_missing_review"
    elif ends:
        status = "end_marker_removed_start_missing_review"
    else:
        status = "markers_absent_review"

    return normalize_newlines(cleaned), status


def apply_text_boundaries(text: str, rules: dict) -> tuple[str, str]:
    """Apply reviewed per-book content boundaries; report missing anchors."""
    start_status = "not_configured"
    end_status = "not_configured"

    def matches(pattern: str) -> list[re.Match]:
        # Edition text often indents headings; allow horizontal indentation
        # before the anchored heading without making the catalog regex noisy.
        pattern = pattern.replace("(?im)^", "(?im)^[ \\t]*")
        return list(re.finditer(pattern, text))

    start_pattern = rules.get("start_pattern")
    if start_pattern:
        found = matches(start_pattern)
        occurrence = rules.get("start_occurrence", 1)
        if occurrence < 1 or occurrence > len(found):
            return text, "start_anchor_missing_review"
        text = text[found[occurrence - 1].start():]
        start_status = "applied"

    end_pattern = rules.get("end_pattern")
    if end_pattern:
        found = matches(end_pattern)
        occurrence = rules.get("end_occurrence", 1)
        if occurrence < 1 or occurrence > len(found):
            return text, f"start={start_status};end_anchor_missing_review"
        text = text[:found[occurrence - 1].start()]
        end_status = "applied"

    return normalize_newlines(text), f"start={start_status};end={end_status}"


def trim_terminal_decoration(text: str) -> str:
    """Remove trailing line-art separators left by some text editions."""
    lines = text.rstrip().splitlines()
    while lines and (
        re.fullmatch(r"[ \t*+|=_~—-]+", lines[-1])
        or lines[-1].strip() == "[Illustration]"
    ):
        lines.pop()
    return "\n".join(lines).strip() + "\n"


def normalize_newlines(text: str) -> str:
    """Standardize line endings and trim outer whitespace without retokenizing."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def find_raw_book(book_id: int) -> Path | None:
    """Find a locally downloaded text whose filename begins with its eBook ID."""
    if not RAW_DIR.exists():
        return None
    matches = sorted(RAW_DIR.glob(f"**/{book_id}_*.txt"))
    return matches[0] if matches else None


def output_name(book: dict) -> str:
    title = re.sub(r"[^a-z0-9]+", "_", book["title"].lower()).strip("_")
    return f"{book['gutenberg_id']}_{title}.txt"


def process_corpus() -> list[dict]:
    """Write one processed file per included book and a status manifest."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Book catalog not found: {CATALOG_PATH}")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    boundaries = json.loads(BOUNDARIES_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for book in catalog:
        author_dir = OUTPUT_DIR / book["author_key"]
        if not book.get("include_for_model", True):
            records.append({
                "gutenberg_id": book["gutenberg_id"],
                "author_key": book["author_key"],
                "title": book["title"],
                "raw_path": "",
                "processed_path": "",
                "cleaning_status": "excluded: " + book["exclusion_reason"],
                "boundary_status": "not_applicable",
                "word_count": 0,
            })
            continue

        raw_path = find_raw_book(book["gutenberg_id"])
        if raw_path is None:
            records.append({
                "gutenberg_id": book["gutenberg_id"],
                "author_key": book["author_key"],
                "title": book["title"],
                "raw_path": "",
                "processed_path": "",
                "cleaning_status": "missing_raw_file",
                "boundary_status": "not_run",
                "word_count": 0,
            })
            print(f"Missing raw text for {book['title']} (ID {book['gutenberg_id']}).")
            continue

        raw_text = raw_path.read_text(encoding="utf-8-sig", errors="replace")
        clean_text, wrapper_status = clean_gutenberg_wrapper(raw_text)
        clean_text, boundary_status = apply_text_boundaries(
            clean_text, boundaries.get(str(book["gutenberg_id"]), {})
        )
        clean_text = trim_terminal_decoration(clean_text)
        book_output_dir = author_dir / "books"
        book_output_dir.mkdir(parents=True, exist_ok=True)
        processed_path = book_output_dir / output_name(book)
        processed_path.write_text(clean_text, encoding="utf-8")
        word_count = len(re.findall(r"\b[\w’'-]+\b", clean_text))

        records.append({
            "gutenberg_id": book["gutenberg_id"],
            "author_key": book["author_key"],
            "title": book["title"],
            "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "processed_path": str(processed_path.relative_to(PROJECT_ROOT)),
            "cleaning_status": wrapper_status,
            "boundary_status": boundary_status,
            "word_count": word_count,
        })
        print(
            f"Processed {book['author_key']}: {book['title']} "
            f"({word_count:,} words; {wrapper_status}; {boundary_status})."
        )

    manifest_path = OUTPUT_DIR / "manifest.csv"
    fieldnames = [
        "gutenberg_id", "author_key", "title", "raw_path",
        "processed_path", "cleaning_status", "boundary_status", "word_count",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote per-book corpus and manifest under {OUTPUT_DIR}.")
    return records


if __name__ == "__main__":
    process_corpus()
