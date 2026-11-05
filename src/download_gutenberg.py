"""Download catalogued Project Gutenberg plain-text editions.

The script is safe to import; downloads only happen when run directly.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CATALOG_PATH = PROJECT_ROOT / "data" / "books.json"
DOWNLOAD_DELAY_SECONDS = 1.0


def safe_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:70]


def raw_path_for(book: dict) -> Path:
    return RAW_DIR / book["author_key"] / (
        f"{book['gutenberg_id']}_{safe_title(book['title'])}.txt"
    )


def candidate_urls(book_id: int) -> list[str]:
    """Try the current cache URL and common legacy plain-text URLs."""
    return [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-8.txt",
    ]


def fetch_book(book: dict, session: requests.Session) -> bool:
    output_path = raw_path_for(book)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(RAW_DIR.glob(f"**/{book['gutenberg_id']}_*.txt"))
    if existing:
        print(f"Already present: {existing[0].relative_to(PROJECT_ROOT)}")
        return True

    for url in candidate_urls(book["gutenberg_id"]):
        try:
            response = session.get(
                url,
                headers={"User-Agent": "WritingStyleMimic/1.0 (educational corpus tool)"},
                timeout=45,
            )
            if response.status_code == 404:
                print(f"Not found at {url}")
                continue
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "text/plain" not in content_type and not url.endswith(".txt"):
                print(f"Skipping non-text response from {url} ({content_type or 'unknown type'}).")
                continue

            text = response.content.decode("utf-8-sig", errors="replace")
            if "<html" in text[:1000].lower() and "text/plain" not in content_type:
                print(f"Skipping HTML response from {url}.")
                continue

            output_path.write_text(text, encoding="utf-8")
            print(f"Downloaded {book['title']} -> {output_path.relative_to(PROJECT_ROOT)}")
            return True
        except requests.RequestException as exc:
            print(f"Download failed at {url}: {exc}")
        finally:
            time.sleep(DOWNLOAD_DELAY_SECONDS)

    print(f"Could not download {book['title']} (eBook {book['gutenberg_id']}).")
    return False


def download_corpus() -> None:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Book catalog not found: {CATALOG_PATH}")

    books = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    session = requests.Session()
    for book in books:
        if not book.get("include_for_model", True):
            print(f"Skipping excluded title: {book['title']} — {book['exclusion_reason']}")
            continue
        fetch_book(book, session)


if __name__ == "__main__":
    download_corpus()
