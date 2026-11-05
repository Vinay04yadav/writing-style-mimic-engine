"""Evaluate a balanced word/character n-gram authorship baseline by book folds."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLDS_PATH = PROJECT_ROOT / "data" / "evaluation" / "book_folds.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "authorship_baseline"

CHUNK_WORDS = 500
CHUNKS_PER_BOOK = 25
SEED = 20261106


def read_chunks(path: Path) -> list[str]:
    """Create equal-length, non-overlapping text chunks from one book."""
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    chunks = [
        " ".join(tokens[start:start + CHUNK_WORDS])
        for start in range(0, len(tokens) - CHUNK_WORDS + 1, CHUNK_WORDS)
    ]
    if len(chunks) < CHUNKS_PER_BOOK:
        raise ValueError(
            f"{path} has only {len(chunks)} complete {CHUNK_WORDS}-token chunks; "
            f"need {CHUNKS_PER_BOOK}."
        )

    rng = random.Random(SEED + int(path.name.split("_", 1)[0]))
    chosen = sorted(rng.sample(range(len(chunks)), CHUNKS_PER_BOOK))
    return [chunks[index] for index in chosen]


def create_vectorizer() -> FeatureUnion:
    """Combine lexical and character-level TF-IDF features."""
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w+\b",
                min_df=2,
                max_features=100_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=300_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
    ])


def score_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def create_confusion_rows(matrix: list[list[int]], labels: list[str]) -> list[dict]:
    return [
        {"true_author": labels[row], **{
            labels[column]: matrix[row][column]
            for column in range(len(labels))
        }}
        for row in range(len(labels))
    ]


def evaluate() -> dict:
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Book folds not found at {FOLDS_PATH}. Run src/build_book_folds.py first."
        )

    with FOLDS_PATH.open(encoding="utf-8", newline="") as folds_file:
        assignments = list(csv.DictReader(folds_file))
    authors = sorted({row["author_key"] for row in assignments})
    fold_ids = sorted({int(row["held_out_fold"]) for row in assignments})

    book_chunks = {}
    for row in assignments:
        book_id = int(row["gutenberg_id"])
        text_path = PROJECT_ROOT / row["processed_path"]
        if not text_path.is_file():
            raise FileNotFoundError(f"Processed text not found: {text_path}")
        book_chunks[book_id] = read_chunks(text_path)

    chunk_true: list[str] = []
    chunk_pred: list[str] = []
    chunk_predictions = []
    book_true: list[str] = []
    book_pred: list[str] = []
    book_predictions = []
    fold_metrics = []

    for fold_id in fold_ids:
        train_rows = [row for row in assignments if int(row["held_out_fold"]) != fold_id]
        test_rows = [row for row in assignments if int(row["held_out_fold"]) == fold_id]

        train_texts = []
        train_labels = []
        for row in train_rows:
            train_texts.extend(book_chunks[int(row["gutenberg_id"])])
            train_labels.extend([row["author_key"]] * CHUNKS_PER_BOOK)

        test_texts = []
        test_labels = []
        test_book_ids = []
        test_chunk_indices = []
        for row in test_rows:
            test_texts.extend(book_chunks[int(row["gutenberg_id"])])
            test_labels.extend([row["author_key"]] * CHUNKS_PER_BOOK)
            test_book_ids.extend([int(row["gutenberg_id"])] * CHUNKS_PER_BOOK)
            test_chunk_indices.extend(range(CHUNKS_PER_BOOK))

        vectorizer = create_vectorizer()
        train_matrix = vectorizer.fit_transform(train_texts)
        test_matrix = vectorizer.transform(test_texts)
        classifier = LinearSVC(
            class_weight="balanced",
            dual="auto",
            random_state=SEED,
        )
        classifier.fit(train_matrix, train_labels)

        scores = classifier.decision_function(test_matrix)
        predictions = classifier.classes_[np.argmax(scores, axis=1)]
        chunk_true.extend(test_labels)
        chunk_pred.extend(predictions.tolist())

        book_scores = defaultdict(list)
        for index, book_id in enumerate(test_book_ids):
            book_scores[book_id].append(scores[index])

        current_book_true = []
        current_book_pred = []
        assignment_by_id = {int(row["gutenberg_id"]): row for row in test_rows}
        for book_id in sorted(book_scores):
            row = assignment_by_id[book_id]
            avg_scores = np.mean(book_scores[book_id], axis=0)
            predicted_author = str(classifier.classes_[int(np.argmax(avg_scores))])
            current_book_true.append(row["author_key"])
            current_book_pred.append(predicted_author)
            book_true.append(row["author_key"])
            book_pred.append(predicted_author)
            book_predictions.append({
                "fold": fold_id,
                "gutenberg_id": book_id,
                "title": row["title"],
                "true_author": row["author_key"],
                "predicted_author": predicted_author,
                "correct": int(predicted_author == row["author_key"]),
            })

        current_test_predictions = predictions.tolist()
        for row_author, predicted_author, book_id, chunk_index in zip(
            test_labels, current_test_predictions, test_book_ids, test_chunk_indices
        ):
            chunk_predictions.append({
                "fold": fold_id,
                "gutenberg_id": book_id,
                "chunk_index": chunk_index,
                "true_author": row_author,
                "predicted_author": predicted_author,
                "correct": int(row_author == predicted_author),
            })

        fold_metrics.append({
            "fold": fold_id,
            "chunk_accuracy": float(accuracy_score(test_labels, predictions)),
            "chunk_macro_f1": float(f1_score(
                test_labels, predictions, labels=authors, average="macro"
            )),
            "book_accuracy": float(accuracy_score(current_book_true, current_book_pred)),
            "book_macro_f1": float(f1_score(
                current_book_true, current_book_pred, labels=authors, average="macro"
            )),
        })
        print(
            f"Fold {fold_id}: chunk accuracy="
            f"{fold_metrics[-1]['chunk_accuracy']:.3f}, book accuracy="
            f"{fold_metrics[-1]['book_accuracy']:.3f}"
        )

    chunk_results = score_metrics(chunk_true, chunk_pred, authors)
    book_results = score_metrics(book_true, book_pred, authors)
    summary = {
        "model": "word + character TF-IDF with LinearSVC",
        "evaluation": "five-fold grouped by complete book; one held-out book per author per fold",
        "seed": SEED,
        "chunk_whitespace_tokens": CHUNK_WORDS,
        "chunks_per_book": CHUNKS_PER_BOOK,
        "included_books": len(assignments),
        "authors": authors,
        "fold_metrics": fold_metrics,
        "out_of_fold_chunk_metrics": {
            key: value for key, value in chunk_results.items() if key != "confusion_matrix"
        },
        "out_of_fold_book_metrics": {
            key: value for key, value in book_results.items() if key != "confusion_matrix"
        },
        "chunk_confusion_matrix": chunk_results["confusion_matrix"],
        "book_confusion_matrix": book_results["confusion_matrix"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        fold_metrics,
        ["fold", "chunk_accuracy", "chunk_macro_f1", "book_accuracy", "book_macro_f1"],
    )
    write_csv(
        OUTPUT_DIR / "book_predictions.csv",
        book_predictions,
        ["fold", "gutenberg_id", "title", "true_author", "predicted_author", "correct"],
    )
    write_csv(
        OUTPUT_DIR / "chunk_predictions.csv",
        chunk_predictions,
        ["fold", "gutenberg_id", "chunk_index", "true_author", "predicted_author", "correct"],
    )
    write_csv(
        OUTPUT_DIR / "book_confusion_matrix.csv",
        create_confusion_rows(book_results["confusion_matrix"], authors),
        ["true_author", *authors],
    )
    write_csv(
        OUTPUT_DIR / "chunk_confusion_matrix.csv",
        create_confusion_rows(chunk_results["confusion_matrix"], authors),
        ["true_author", *authors],
    )

    print(
        f"Out-of-fold chunks: accuracy={chunk_results['accuracy']:.3f}, "
        f"macro-F1={chunk_results['macro_f1']:.3f}"
    )
    print(
        f"Held-out books: accuracy={book_results['accuracy']:.3f}, "
        f"macro-F1={book_results['macro_f1']:.3f}"
    )
    print(f"Saved metrics and predictions under {OUTPUT_DIR}.")
    return summary


if __name__ == "__main__":
    evaluate()
