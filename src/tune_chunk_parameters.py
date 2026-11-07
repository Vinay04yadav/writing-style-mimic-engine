"""Compare chunk sizes and per-book sample counts with the leading baseline.

Uses the existing five book-held-out folds and combined TF-IDF + LinearSVC.
For each chunk size, sampled chunk indices are nested as the requested count
increases, so count comparisons reuse the same passages where possible.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from evaluate_authorship import (
    FOLDS_PATH,
    PROJECT_ROOT,
    SEED,
    create_vectorizer,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "chunk_parameter_comparison"
CONFIGURATIONS = [
    (250, 10),
    (250, 20),
    (250, 40),
    (500, 5),
    (500, 10),
    (500, 20),
    (500, 25),
    (500, 34),
    (1000, 5),
    (1000, 10),
    (1000, 17),
]


def load_book_chunks(path: Path, book_id: int, chunk_words: int, max_count: int):
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    chunks = [
        " ".join(tokens[start:start + chunk_words])
        for start in range(0, len(tokens) - chunk_words + 1, chunk_words)
    ]
    if len(chunks) < max_count:
        raise ValueError(
            f"{path} has {len(chunks)} complete chunks of {chunk_words} tokens; "
            f"need at least {max_count}."
        )
    rng = random.Random(SEED + book_id + chunk_words)
    chosen_indices = rng.sample(range(len(chunks)), max_count)
    return [(index, chunks[index]) for index in chosen_indices]


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate() -> list[dict]:
    with FOLDS_PATH.open(encoding="utf-8", newline="") as folds_file:
        assignments = list(csv.DictReader(folds_file))
    authors = sorted({row["author_key"] for row in assignments})
    fold_ids = sorted({int(row["held_out_fold"]) for row in assignments})
    needed_by_size = defaultdict(int)
    for chunk_words, count in CONFIGURATIONS:
        needed_by_size[chunk_words] = max(needed_by_size[chunk_words], count)

    sampled_by_size = {}
    row_by_id = {}
    for row in assignments:
        book_id = int(row["gutenberg_id"])
        row_by_id[book_id] = row
        path = PROJECT_ROOT / row["processed_path"]
        for chunk_words, max_count in needed_by_size.items():
            sampled_by_size[(book_id, chunk_words)] = load_book_chunks(
                path, book_id, chunk_words, max_count
            )

    all_fold_rows = []
    all_book_predictions = []
    summary_rows = []

    for chunk_words, chunks_per_book in CONFIGURATIONS:
        out_true, out_pred = [], []
        out_book_true, out_book_pred = [], []
        config_fold_rows = []
        config_book_predictions = []

        for fold_id in fold_ids:
            train_rows = [r for r in assignments if int(r["held_out_fold"]) != fold_id]
            test_rows = [r for r in assignments if int(r["held_out_fold"]) == fold_id]
            train_texts, train_labels = [], []
            test_texts, test_labels, test_book_ids = [], [], []
            for row in train_rows:
                book_id = int(row["gutenberg_id"])
                chunks = sampled_by_size[(book_id, chunk_words)][:chunks_per_book]
                train_texts.extend(text for _, text in chunks)
                train_labels.extend([row["author_key"]] * chunks_per_book)
            for row in test_rows:
                book_id = int(row["gutenberg_id"])
                chunks = sampled_by_size[(book_id, chunk_words)][:chunks_per_book]
                test_texts.extend(text for _, text in chunks)
                test_labels.extend([row["author_key"]] * chunks_per_book)
                test_book_ids.extend([book_id] * chunks_per_book)

            vectorizer = create_vectorizer()
            train_matrix = vectorizer.fit_transform(train_texts)
            test_matrix = vectorizer.transform(test_texts)
            classifier = LinearSVC(
                class_weight="balanced", dual="auto", random_state=SEED
            )
            classifier.fit(train_matrix, train_labels)
            scores = classifier.decision_function(test_matrix)
            chunk_pred = classifier.classes_[np.argmax(scores, axis=1)].tolist()
            out_true.extend(test_labels)
            out_pred.extend(chunk_pred)

            by_book = defaultdict(list)
            for index, book_id in enumerate(test_book_ids):
                by_book[book_id].append(scores[index])
            fold_book_true, fold_book_pred = [], []
            for book_id in sorted(by_book):
                row = row_by_id[book_id]
                predicted = str(classifier.classes_[
                    int(np.argmax(np.mean(by_book[book_id], axis=0)))
                ])
                fold_book_true.append(row["author_key"])
                fold_book_pred.append(predicted)
                out_book_true.append(row["author_key"])
                out_book_pred.append(predicted)
                prediction = {
                    "chunk_words": chunk_words,
                    "chunks_per_book": chunks_per_book,
                    "fold": fold_id,
                    "gutenberg_id": book_id,
                    "title": row["title"],
                    "true_author": row["author_key"],
                    "predicted_author": predicted,
                    "correct": int(predicted == row["author_key"]),
                }
                config_book_predictions.append(prediction)
                all_book_predictions.append(prediction)

            fold_metric = {
                "chunk_words": chunk_words,
                "chunks_per_book": chunks_per_book,
                "fold": fold_id,
                "chunk_accuracy": accuracy_score(test_labels, chunk_pred),
                "chunk_macro_f1": f1_score(
                    test_labels, chunk_pred, labels=authors, average="macro"
                ),
                "book_accuracy": accuracy_score(fold_book_true, fold_book_pred),
                "book_macro_f1": f1_score(
                    fold_book_true, fold_book_pred, labels=authors, average="macro"
                ),
            }
            config_fold_rows.append(fold_metric)
            all_fold_rows.append(fold_metric)

        summary_rows.append({
            "chunk_words": chunk_words,
            "chunks_per_book": chunks_per_book,
            "tokens_per_book_sampled": chunk_words * chunks_per_book,
            "chunk_accuracy": accuracy_score(out_true, out_pred),
            "chunk_macro_f1": f1_score(out_true, out_pred, labels=authors, average="macro"),
            "book_accuracy": accuracy_score(out_book_true, out_book_pred),
            "book_macro_f1": f1_score(
                out_book_true, out_book_pred, labels=authors, average="macro"
            ),
            "fold_chunk_accuracy_min": min(r["chunk_accuracy"] for r in config_fold_rows),
            "fold_chunk_accuracy_max": max(r["chunk_accuracy"] for r in config_fold_rows),
        })
        print(
            f"{chunk_words} tokens x {chunks_per_book} chunks: "
            f"chunk F1={summary_rows[-1]['chunk_macro_f1']:.4f}, "
            f"book F1={summary_rows[-1]['book_macro_f1']:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT_DIR / "summary.csv", summary_rows, list(summary_rows[0]))
    write_rows(OUTPUT_DIR / "fold_metrics.csv", all_fold_rows, list(all_fold_rows[0]))
    write_rows(
        OUTPUT_DIR / "book_predictions.csv",
        all_book_predictions,
        list(all_book_predictions[0]),
    )
    return summary_rows


if __name__ == "__main__":
    evaluate()
