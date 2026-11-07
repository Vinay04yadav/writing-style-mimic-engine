"""Measure chunk-sampling variability and locate recurring attribution errors."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from evaluate_authorship import FOLDS_PATH, PROJECT_ROOT, create_vectorizer


OUTPUT_DIR = PROJECT_ROOT / "results" / "authorship_robustness"
SAMPLE_SEEDS = [20261106, 42, 2026, 7, 8675309]
CONFIGURATIONS = [(500, 25), (1000, 10)]
MODEL_SEED = 20261106


def load_sampled_chunks(
    path: Path,
    book_id: int,
    chunk_words: int,
    chunks_per_book: int,
    sample_seed: int,
) -> list[tuple[int, str]]:
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    chunks = [
        " ".join(tokens[start:start + chunk_words])
        for start in range(0, len(tokens) - chunk_words + 1, chunk_words)
    ]
    if len(chunks) < chunks_per_book:
        raise ValueError(
            f"{path} has {len(chunks)} complete chunks of {chunk_words} tokens; "
            f"need {chunks_per_book}."
        )
    # Matches evaluate_authorship.py when sample_seed is the original project
    # seed and chunk_words is 500; larger chunk sizes use the same seed rule.
    rng = random.Random(sample_seed + book_id)
    indices = rng.sample(range(len(chunks)), chunks_per_book)
    return [(index, chunks[index]) for index in indices]


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_configuration(assignments, authors, fold_ids, chunk_words, chunks_per_book):
    seed_metrics = []
    all_chunk_predictions = []
    all_book_predictions = []
    assignment_by_id = {int(row["gutenberg_id"]): row for row in assignments}

    for sample_seed in SAMPLE_SEEDS:
        sampled = {}
        for row in assignments:
            book_id = int(row["gutenberg_id"])
            sampled[book_id] = load_sampled_chunks(
                PROJECT_ROOT / row["processed_path"],
                book_id,
                chunk_words,
                chunks_per_book,
                sample_seed,
            )

        seed_chunk_predictions = []
        seed_book_predictions = []
        for fold_id in fold_ids:
            train_rows = [r for r in assignments if int(r["held_out_fold"]) != fold_id]
            test_rows = [r for r in assignments if int(r["held_out_fold"]) == fold_id]
            train_texts, train_labels = [], []
            test_texts, test_labels, test_book_ids, test_indices = [], [], [], []

            for row in train_rows:
                book_id = int(row["gutenberg_id"])
                selected = sampled[book_id]
                train_texts.extend(text for _, text in selected)
                train_labels.extend([row["author_key"]] * chunks_per_book)
            for row in test_rows:
                book_id = int(row["gutenberg_id"])
                selected = sampled[book_id]
                test_texts.extend(text for _, text in selected)
                test_labels.extend([row["author_key"]] * chunks_per_book)
                test_book_ids.extend([book_id] * chunks_per_book)
                test_indices.extend(index for index, _ in selected)

            vectorizer = create_vectorizer()
            train_matrix = vectorizer.fit_transform(train_texts)
            test_matrix = vectorizer.transform(test_texts)
            classifier = LinearSVC(
                class_weight="balanced", dual="auto", random_state=MODEL_SEED
            )
            classifier.fit(train_matrix, train_labels)
            scores = classifier.decision_function(test_matrix)
            predictions = classifier.classes_[np.argmax(scores, axis=1)].tolist()

            for true_author, predicted_author, book_id, chunk_index in zip(
                test_labels, predictions, test_book_ids, test_indices
            ):
                row = assignment_by_id[book_id]
                prediction = {
                    "chunk_words": chunk_words,
                    "chunks_per_book": chunks_per_book,
                    "sample_seed": sample_seed,
                    "fold": fold_id,
                    "gutenberg_id": book_id,
                    "title": row["title"],
                    "true_author": true_author,
                    "predicted_author": predicted_author,
                    "correct": int(true_author == predicted_author),
                    "chunk_index": chunk_index,
                }
                seed_chunk_predictions.append(prediction)
                all_chunk_predictions.append(prediction)

            scores_by_book = defaultdict(list)
            for index, book_id in enumerate(test_book_ids):
                scores_by_book[book_id].append(scores[index])
            for book_id, book_scores in scores_by_book.items():
                row = assignment_by_id[book_id]
                average_scores = np.mean(book_scores, axis=0)
                predicted_author = str(
                    classifier.classes_[int(np.argmax(average_scores))]
                )
                prediction = {
                    "chunk_words": chunk_words,
                    "chunks_per_book": chunks_per_book,
                    "sample_seed": sample_seed,
                    "fold": fold_id,
                    "gutenberg_id": book_id,
                    "title": row["title"],
                    "true_author": row["author_key"],
                    "predicted_author": predicted_author,
                    "correct": int(row["author_key"] == predicted_author),
                }
                seed_book_predictions.append(prediction)
                all_book_predictions.append(prediction)

        y_true = [r["true_author"] for r in seed_chunk_predictions]
        y_pred = [r["predicted_author"] for r in seed_chunk_predictions]
        book_true = [r["true_author"] for r in seed_book_predictions]
        book_pred = [r["predicted_author"] for r in seed_book_predictions]
        seed_metrics.append({
            "chunk_words": chunk_words,
            "chunks_per_book": chunks_per_book,
            "sample_seed": sample_seed,
            "chunk_accuracy": accuracy_score(y_true, y_pred),
            "chunk_macro_f1": f1_score(y_true, y_pred, labels=authors, average="macro"),
            "book_accuracy": accuracy_score(book_true, book_pred),
            "book_macro_f1": f1_score(
                book_true, book_pred, labels=authors, average="macro"
            ),
            "chunk_errors": sum(r["correct"] == 0 for r in seed_chunk_predictions),
            "book_errors": sum(r["correct"] == 0 for r in seed_book_predictions),
        })
        print(
            f"{chunk_words} x {chunks_per_book}, seed {sample_seed}: "
            f"chunk accuracy={seed_metrics[-1]['chunk_accuracy']:.4f}, "
            f"macro-F1={seed_metrics[-1]['chunk_macro_f1']:.4f}, "
            f"book accuracy={seed_metrics[-1]['book_accuracy']:.4f}"
        )

    all_y_true = [r["true_author"] for r in all_chunk_predictions]
    all_y_pred = [r["predicted_author"] for r in all_chunk_predictions]
    aggregate = {
        "chunk_words": chunk_words,
        "chunks_per_book": chunks_per_book,
        "sample_seeds": len(SAMPLE_SEEDS),
        "chunk_accuracy_mean": np.mean([r["chunk_accuracy"] for r in seed_metrics]),
        "chunk_accuracy_std": np.std([r["chunk_accuracy"] for r in seed_metrics]),
        "chunk_macro_f1_mean": np.mean([r["chunk_macro_f1"] for r in seed_metrics]),
        "chunk_macro_f1_std": np.std([r["chunk_macro_f1"] for r in seed_metrics]),
        "book_accuracy_mean": np.mean([r["book_accuracy"] for r in seed_metrics]),
        "book_accuracy_std": np.std([r["book_accuracy"] for r in seed_metrics]),
        "pooled_chunk_accuracy": accuracy_score(all_y_true, all_y_pred),
        "pooled_chunk_macro_f1": f1_score(
            all_y_true, all_y_pred, labels=authors, average="macro"
        ),
        "min_seed_chunk_accuracy": min(r["chunk_accuracy"] for r in seed_metrics),
        "max_seed_chunk_accuracy": max(r["chunk_accuracy"] for r in seed_metrics),
        "total_chunk_errors": sum(r["chunk_errors"] for r in seed_metrics),
        "total_book_errors": sum(r["book_errors"] for r in seed_metrics),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=authors, zero_division=0
    )
    author_metrics = [
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
    matrix = confusion_matrix(all_y_true, all_y_pred, labels=authors)
    confusion_rows = [
        {
            "chunk_words": chunk_words,
            "chunks_per_book": chunks_per_book,
            "true_author": authors[i],
            **{authors[j]: int(matrix[i, j]) for j in range(len(authors))},
        }
        for i in range(len(authors))
    ]

    per_book = defaultdict(list)
    for row in all_chunk_predictions:
        per_book[(row["gutenberg_id"], row["title"], row["true_author"])].append(row)
    book_error_rows = []
    for (book_id, title, author), records in sorted(per_book.items()):
        errors = sum(record["correct"] == 0 for record in records)
        book_error_rows.append({
            "chunk_words": chunk_words,
            "chunks_per_book": chunks_per_book,
            "gutenberg_id": book_id,
            "title": title,
            "author": author,
            "tested_chunks_across_seeds": len(records),
            "chunk_errors": errors,
            "chunk_error_rate": errors / len(records),
            "seeds_with_at_least_one_error": len({
                record["sample_seed"] for record in records if not record["correct"]
            }),
        })

    return seed_metrics, aggregate, all_chunk_predictions, all_book_predictions, author_metrics, confusion_rows, book_error_rows


def main() -> None:
    with FOLDS_PATH.open(encoding="utf-8", newline="") as folds_file:
        assignments = list(csv.DictReader(folds_file))
    authors = sorted({row["author_key"] for row in assignments})
    fold_ids = sorted({int(row["held_out_fold"]) for row in assignments})

    all_seeds, all_aggregates = [], []
    all_chunks, all_books = [], []
    all_authors, all_confusions, all_book_errors = [], [], []
    for chunk_words, count in CONFIGURATIONS:
        outputs = run_configuration(
            assignments, authors, fold_ids, chunk_words, count
        )
        seed_rows, aggregate, chunks, books, author_rows, confusion_rows, book_error_rows = outputs
        all_seeds.extend(seed_rows)
        all_aggregates.append(aggregate)
        all_chunks.extend(chunks)
        all_books.extend(books)
        all_authors.extend(author_rows)
        all_confusions.extend(confusion_rows)
        all_book_errors.extend(book_error_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT_DIR / "seed_metrics.csv", all_seeds, list(all_seeds[0]))
    write_rows(OUTPUT_DIR / "aggregate_summary.csv", all_aggregates, list(all_aggregates[0]))
    write_rows(OUTPUT_DIR / "chunk_predictions.csv", all_chunks, list(all_chunks[0]))
    write_rows(OUTPUT_DIR / "book_predictions.csv", all_books, list(all_books[0]))
    write_rows(OUTPUT_DIR / "author_metrics.csv", all_authors, list(all_authors[0]))
    write_rows(OUTPUT_DIR / "confusion_matrices.csv", all_confusions, list(all_confusions[0]))
    write_rows(OUTPUT_DIR / "book_error_summary.csv", all_book_errors, list(all_book_errors[0]))

    print("\nMean and standard deviation across sampling seeds:")
    for row in all_aggregates:
        print(
            f"{row['chunk_words']} x {row['chunks_per_book']}: "
            f"chunk accuracy={row['chunk_accuracy_mean']:.4f} "
            f"(+/- {row['chunk_accuracy_std']:.4f}), "
            f"macro-F1={row['chunk_macro_f1_mean']:.4f} "
            f"(+/- {row['chunk_macro_f1_std']:.4f}), "
            f"book accuracy={row['book_accuracy_mean']:.4f}, "
            f"chunk errors={row['total_chunk_errors']}"
        )
    print(f"Saved robustness analyses under {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
