"""Test the fixed authorship baseline on books excluded from CV training."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from evaluate_authorship import (
    FOLDS_PATH,
    PROJECT_ROOT,
    create_vectorizer,
)


CATALOG_PATH = PROJECT_ROOT / "data" / "books.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed_books" / "manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "external_validation"
SAMPLE_SEED = 20261106
MODEL_SEED = 20261106
CONFIGURATIONS = [(500, 25), (1000, 10)]


def sampled_chunks(path: Path, book_id: int, chunk_words: int, count: int):
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    chunks = [
        " ".join(tokens[start:start + chunk_words])
        for start in range(0, len(tokens) - chunk_words + 1, chunk_words)
    ]
    if len(chunks) < count:
        raise ValueError(
            f"{path} has {len(chunks)} full chunks of {chunk_words} tokens; "
            f"need {count}."
        )
    rng = random.Random(SAMPLE_SEED + book_id)
    indices = rng.sample(range(len(chunks)), count)
    return [(index, chunks[index]) for index in indices]


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_inputs():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    with FOLDS_PATH.open(encoding="utf-8", newline="") as folds_file:
        training_rows = list(csv.DictReader(folds_file))
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as manifest_file:
        manifest = {
            int(row["gutenberg_id"]): row
            for row in csv.DictReader(manifest_file)
        }

    external_books = [
        book for book in catalog
        if book.get("include_for_model", True)
        and book.get("evaluation_group") == "external_validation"
    ]
    training_ids = {int(row["gutenberg_id"]) for row in training_rows}
    if training_ids.intersection(int(book["gutenberg_id"]) for book in external_books):
        raise ValueError("External validation books overlap the training folds.")

    training_authors = {row["author_key"] for row in training_rows}
    external_authors = {book["author_key"] for book in external_books}
    if training_authors != external_authors:
        raise ValueError(
            f"External authors {sorted(external_authors)} do not match "
            f"training authors {sorted(training_authors)}."
        )
    counts = defaultdict(int)
    for book in external_books:
        counts[book["author_key"]] += 1
        record = manifest.get(int(book["gutenberg_id"]))
        if record is None or not record["processed_path"]:
            raise ValueError(f"No processed text for {book['title']}.")
        if not record["boundary_status"].startswith("start=applied"):
            raise ValueError(
                f"Content start boundary missing for {book['title']}: "
                f"{record['boundary_status']}"
            )
        book["processed_path"] = record["processed_path"]
        book["word_count"] = int(record["word_count"])
    if any(count != 2 for count in counts.values()):
        raise ValueError(f"Expected two external books per author; found {dict(counts)}.")
    return training_rows, external_books


def evaluate_configuration(training_rows, external_books, chunk_words, chunks_per_book):
    train_texts, train_labels = [], []
    test_texts, test_labels, test_book_ids, test_chunk_indices = [], [], [], []
    for row in training_rows:
        book_id = int(row["gutenberg_id"])
        chunks = sampled_chunks(
            PROJECT_ROOT / row["processed_path"], book_id, chunk_words, chunks_per_book
        )
        train_texts.extend(text for _, text in chunks)
        train_labels.extend([row["author_key"]] * chunks_per_book)
    for book in external_books:
        book_id = int(book["gutenberg_id"])
        chunks = sampled_chunks(
            PROJECT_ROOT / book["processed_path"], book_id, chunk_words, chunks_per_book
        )
        test_texts.extend(text for _, text in chunks)
        test_labels.extend([book["author_key"]] * chunks_per_book)
        test_book_ids.extend([book_id] * chunks_per_book)
        test_chunk_indices.extend(index for index, _ in chunks)

    vectorizer = create_vectorizer()
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    classifier = LinearSVC(
        class_weight="balanced", dual="auto", random_state=MODEL_SEED
    )
    classifier.fit(train_matrix, train_labels)
    scores = classifier.decision_function(test_matrix)
    chunk_predictions = classifier.classes_[np.argmax(scores, axis=1)].tolist()
    authors = sorted(set(train_labels))

    chunk_rows = []
    assignment_by_id = {int(book["gutenberg_id"]): book for book in external_books}
    scores_by_book = defaultdict(list)
    for index, (true_author, predicted_author, book_id, chunk_index) in enumerate(zip(
        test_labels, chunk_predictions, test_book_ids, test_chunk_indices
    )):
        book = assignment_by_id[book_id]
        chunk_rows.append({
            "chunk_words": chunk_words,
            "chunks_per_book": chunks_per_book,
            "gutenberg_id": book_id,
            "title": book["title"],
            "chunk_index": chunk_index,
            "true_author": true_author,
            "predicted_author": predicted_author,
            "correct": int(true_author == predicted_author),
        })
        scores_by_book[book_id].append(scores[index])

    book_rows = []
    for book_id in sorted(scores_by_book):
        book = assignment_by_id[book_id]
        mean_scores = np.mean(scores_by_book[book_id], axis=0)
        predicted_author = str(classifier.classes_[int(np.argmax(mean_scores))])
        book_rows.append({
            "chunk_words": chunk_words,
            "chunks_per_book": chunks_per_book,
            "gutenberg_id": book_id,
            "title": book["title"],
            "true_author": book["author_key"],
            "predicted_author": predicted_author,
            "correct": int(book["author_key"] == predicted_author),
        })

    book_true = [row["true_author"] for row in book_rows]
    book_pred = [row["predicted_author"] for row in book_rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, chunk_predictions, labels=authors, zero_division=0
    )
    author_rows = [
        {
            "chunk_words": chunk_words,
            "chunks_per_book": chunks_per_book,
            "author": author,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
        }
        for i, author in enumerate(authors)
    ]
    matrix = confusion_matrix(test_labels, chunk_predictions, labels=authors)
    confusion_rows = [
        {
            "chunk_words": chunk_words,
            "chunks_per_book": chunks_per_book,
            "true_author": authors[i],
            **{authors[j]: int(matrix[i, j]) for j in range(len(authors))},
        }
        for i in range(len(authors))
    ]
    summary = {
        "chunk_words": chunk_words,
        "chunks_per_book": chunks_per_book,
        "training_books": len(training_rows),
        "external_test_books": len(external_books),
        "chunk_accuracy": accuracy_score(test_labels, chunk_predictions),
        "chunk_macro_f1": f1_score(
            test_labels, chunk_predictions, labels=authors, average="macro"
        ),
        "book_accuracy": accuracy_score(book_true, book_pred),
        "book_macro_f1": f1_score(book_true, book_pred, labels=authors, average="macro"),
        "chunk_errors": sum(not row["correct"] for row in chunk_rows),
        "book_errors": sum(not row["correct"] for row in book_rows),
    }
    return summary, chunk_rows, book_rows, author_rows, confusion_rows


def main() -> None:
    training_rows, external_books = load_inputs()
    outputs = [
        evaluate_configuration(training_rows, external_books, chunk_words, count)
        for chunk_words, count in CONFIGURATIONS
    ]
    summaries = [output[0] for output in outputs]
    chunk_rows = [row for output in outputs for row in output[1]]
    book_rows = [row for output in outputs for row in output[2]]
    author_rows = [row for output in outputs for row in output[3]]
    confusion_rows = [row for output in outputs for row in output[4]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT_DIR / "summary.csv", summaries, list(summaries[0]))
    write_rows(OUTPUT_DIR / "chunk_predictions.csv", chunk_rows, list(chunk_rows[0]))
    write_rows(OUTPUT_DIR / "book_predictions.csv", book_rows, list(book_rows[0]))
    write_rows(OUTPUT_DIR / "author_metrics.csv", author_rows, list(author_rows[0]))
    write_rows(
        OUTPUT_DIR / "confusion_matrices.csv", confusion_rows, list(confusion_rows[0])
    )

    for row in summaries:
        print(
            f"{row['chunk_words']} x {row['chunks_per_book']}: "
            f"chunk accuracy={row['chunk_accuracy']:.4f}, "
            f"macro-F1={row['chunk_macro_f1']:.4f}; "
            f"book accuracy={row['book_accuracy']:.4f} "
            f"({row['external_test_books']} new books)"
        )
    print(f"Saved external evaluation results under {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
