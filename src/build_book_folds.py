"""Create a deterministic five-fold, book-level evaluation assignment."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "books.json"
PROCESSED_MANIFEST = PROJECT_ROOT / "data" / "processed_books" / "manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "book_folds.csv"
N_SPLITS = 5
SEED = 20261106


def create_book_folds() -> list[dict]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    with PROCESSED_MANIFEST.open(encoding="utf-8", newline="") as manifest_file:
        processed = {
            int(row["gutenberg_id"]): row
            for row in csv.DictReader(manifest_file)
        }

    included = [
        book for book in catalog
        if book.get("include_for_model", True)
        and book.get("evaluation_group", "cross_validation") == "cross_validation"
    ]
    authors = sorted({book["author_key"] for book in included})
    rows = []

    for author_index, author_key in enumerate(authors):
        books = sorted(
            (book for book in included if book["author_key"] == author_key),
            key=lambda book: int(book["gutenberg_id"]),
        )
        if len(books) != N_SPLITS:
            raise ValueError(
                f"Expected {N_SPLITS} books for {author_key}; found {len(books)}."
            )

        rng = random.Random(SEED + author_index)
        rng.shuffle(books)
        for held_out_fold, book in enumerate(books, start=1):
            book_id = int(book["gutenberg_id"])
            record = processed.get(book_id)
            if record is None or not record["processed_path"]:
                raise ValueError(f"No processed book found for eBook {book_id}.")
            if not record["boundary_status"].startswith("start=applied"):
                raise ValueError(
                    f"Content start boundary was not applied for {book['title']}: "
                    f"{record['boundary_status']}"
                )

            rows.append({
                "held_out_fold": held_out_fold,
                "gutenberg_id": book_id,
                "author_key": author_key,
                "author": book["author"],
                "title": book["title"],
                "word_count": int(record["word_count"]),
                "processed_path": record["processed_path"],
                "source_url": book["source_url"],
            })

    rows.sort(key=lambda row: (row["held_out_fold"], row["author_key"]))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "held_out_fold", "gutenberg_id", "author_key", "author", "title",
        "word_count", "processed_path", "source_url",
    ]
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} book assignments to {OUTPUT_PATH}.")
    print(
        f"Each of {N_SPLITS} folds holds out one complete book per author; "
        "all other books are training data for that fold."
    )
    return rows


if __name__ == "__main__":
    create_book_folds()
