"""Compare TF-IDF feature sets and linear text classifiers on fixed book folds.

This is a controlled follow-up to evaluate_authorship.py: every configuration
uses the same held-out books and the same 25 sampled chunks per book.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from evaluate_authorship import (
    CHUNKS_PER_BOOK,
    FOLDS_PATH,
    PROJECT_ROOT,
    SEED,
    read_chunks,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "authorship_comparison"


def create_vectorizers() -> dict[str, TfidfVectorizer]:
    common = {
        "min_df": 2,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    }
    return {
        "word_only": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            max_features=100_000,
            **common,
        ),
        "char_only": TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=300_000,
            **common,
        ),
    }


def create_classifiers() -> dict[str, object]:
    return {
        "LinearSVC": LinearSVC(
            class_weight="balanced", dual="auto", random_state=SEED
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        ),
        "MultinomialNB": MultinomialNB(alpha=1.0),
    }


def evaluate_comparison() -> list[dict]:
    with FOLDS_PATH.open(encoding="utf-8", newline="") as folds_file:
        assignments = list(csv.DictReader(folds_file))
    author_labels = sorted({row["author_key"] for row in assignments})
    fold_ids = sorted({int(row["held_out_fold"]) for row in assignments})
    chunks_by_book = {}
    rows_by_book = {}
    for row in assignments:
        book_id = int(row["gutenberg_id"])
        chunks_by_book[book_id] = read_chunks(
            PROJECT_ROOT / row["processed_path"]
        )
        rows_by_book[book_id] = row

    # Aggregate out-of-fold predictions separately for every comparison.
    predictions = {
        (feature_set, model_name): {"true": [], "pred": [], "book_true": [], "book_pred": []}
        for feature_set in ("word_only", "char_only", "word_plus_char")
        for model_name in create_classifiers()
    }
    fold_rows = []

    for fold_id in fold_ids:
        train_rows = [row for row in assignments if int(row["held_out_fold"]) != fold_id]
        test_rows = [row for row in assignments if int(row["held_out_fold"]) == fold_id]
        train_texts = []
        train_labels = []
        test_texts = []
        test_labels = []
        test_book_ids = []
        for row in train_rows:
            chunks = chunks_by_book[int(row["gutenberg_id"])]
            train_texts.extend(chunks)
            train_labels.extend([row["author_key"]] * CHUNKS_PER_BOOK)
        for row in test_rows:
            book_id = int(row["gutenberg_id"])
            chunks = chunks_by_book[book_id]
            test_texts.extend(chunks)
            test_labels.extend([row["author_key"]] * CHUNKS_PER_BOOK)
            test_book_ids.extend([book_id] * CHUNKS_PER_BOOK)

        word_vectorizer, char_vectorizer = create_vectorizers().values()
        train_word = word_vectorizer.fit_transform(train_texts)
        test_word = word_vectorizer.transform(test_texts)
        train_char = char_vectorizer.fit_transform(train_texts)
        test_char = char_vectorizer.transform(test_texts)
        feature_matrices = {
            "word_only": (train_word, test_word),
            "char_only": (train_char, test_char),
            "word_plus_char": (
                hstack((train_word, train_char), format="csr"),
                hstack((test_word, test_char), format="csr"),
            ),
        }

        for feature_set, (train_matrix, test_matrix) in feature_matrices.items():
            for model_name, classifier in create_classifiers().items():
                classifier.fit(train_matrix, train_labels)
                chunk_pred = classifier.predict(test_matrix).tolist()
                record = predictions[(feature_set, model_name)]
                record["true"].extend(test_labels)
                record["pred"].extend(chunk_pred)

                if hasattr(classifier, "decision_function"):
                    scores = classifier.decision_function(test_matrix)
                else:
                    scores = classifier.predict_log_proba(test_matrix)
                per_book_scores = defaultdict(list)
                for index, book_id in enumerate(test_book_ids):
                    per_book_scores[book_id].append(scores[index])
                for book_id, score_list in per_book_scores.items():
                    avg_scores = np.mean(score_list, axis=0)
                    predicted_author = str(classifier.classes_[int(np.argmax(avg_scores))])
                    true_author = rows_by_book[book_id]["author_key"]
                    record["book_true"].append(true_author)
                    record["book_pred"].append(predicted_author)

                fold_rows.append({
                    "fold": fold_id,
                    "feature_set": feature_set,
                    "classifier": model_name,
                    "chunk_accuracy": accuracy_score(test_labels, chunk_pred),
                    "chunk_macro_f1": f1_score(
                        test_labels, chunk_pred, labels=author_labels, average="macro"
                    ),
                    "book_accuracy": accuracy_score(
                        record["book_true"][-len(test_rows):],
                        record["book_pred"][-len(test_rows):],
                    ),
                    "book_macro_f1": f1_score(
                        record["book_true"][-len(test_rows):],
                        record["book_pred"][-len(test_rows):],
                        labels=author_labels,
                        average="macro",
                    ),
                })
        print(f"Completed fold {fold_id} across 9 configurations.")

    summary_rows = []
    for key, record in predictions.items():
        feature_set, model_name = key
        matching_folds = [
            row for row in fold_rows
            if row["feature_set"] == feature_set and row["classifier"] == model_name
        ]
        summary_rows.append({
            "feature_set": feature_set,
            "classifier": model_name,
            "chunk_accuracy": accuracy_score(record["true"], record["pred"]),
            "chunk_macro_f1": f1_score(
                record["true"], record["pred"], labels=author_labels, average="macro"
            ),
            "book_accuracy": accuracy_score(record["book_true"], record["book_pred"]),
            "book_macro_f1": f1_score(
                record["book_true"], record["book_pred"], labels=author_labels, average="macro"
            ),
            "fold_chunk_accuracy_min": min(row["chunk_accuracy"] for row in matching_folds),
            "fold_chunk_accuracy_max": max(row["chunk_accuracy"] for row in matching_folds),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(summary_rows[0])
    for filename, rows, fieldnames in (
        ("summary.csv", summary_rows, fields),
        ("fold_metrics.csv", fold_rows, list(fold_rows[0])),
    ):
        with (OUTPUT_DIR / filename).open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    summary_rows.sort(key=lambda row: (-row["chunk_macro_f1"], -row["book_macro_f1"]))
    print("\nRanked by out-of-fold chunk macro-F1:")
    for row in summary_rows:
        print(
            f"{row['feature_set']:14s} {row['classifier']:20s} "
            f"chunk F1={row['chunk_macro_f1']:.4f} "
            f"book F1={row['book_macro_f1']:.4f}"
        )
    return summary_rows


if __name__ == "__main__":
    evaluate_comparison()
