# Writing Style Mimic Engine

## Overview

This project explores two tasks: identifying an author's writing from a passage and generating new text with author-conditioned style. The current milestone improves the corpus and establishes a reproducible authorship baseline; generation remains experimental.

## Features

- Catalogued corpus of five authors with book-level provenance and reviewable text boundaries.
- Deterministic grouped evaluation folds that hold out complete books.
- Word and character TF-IDF authorship baseline with LinearSVC.
- Historical GPT-2 fine-tuning and generation experiments in a Colab notebook.

## Project Structure

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [Your Repo URL Here]
    cd Writing_Style_Mimic
    ```
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data** only if running the historical notebook:
    ```bash
    python -m nltk.downloader punkt averaged_perceptron_tagger
    ```

## Usage

1.  **Download Data:** Modify the `BOOKS_TO_DOWNLOAD` list in `src/download_gutenberg.py` with desired authors/book IDs from Project Gutenberg. Then run:
    ```bash
    python src/download_gutenberg.py
    ```
    _(This populates `data/raw`)_
2.  **Preprocess Data:** From the project root, run the preprocessing script:
    ```bash
    python src/preprocess_corpus.py
    ```
    _(This writes one file per book under `data/processed_books/<author>/books/` and a `manifest.csv`. It strips standard Gutenberg wrappers when present and applies the reviewed per-book body boundaries in `data/text_boundaries.json`; raw files remain unchanged.)_
3.  **Build Evaluation Folds:** After preprocessing, run:
    ```bash
    python src/build_book_folds.py
    ```
    This writes `data/evaluation/book_folds.csv`. It uses seed `20261106` and assigns one complete book from each author to each of five held-out folds. For fold *k*, rows with `held_out_fold = k` are test books; every other included book is training data. Each of the 25 books appears in the test set exactly once.
4.  **Run Authorship Baseline:**
    ```bash
    python src/evaluate_authorship.py
    ```
    Results are written to `results/authorship_baseline/`, including fold metrics, chunk and book predictions, confusion matrices, and a JSON summary.
5.  **Review Generation Experiments:** The available notebook is `notebooks/NLP_MP_Gem.ipynb`. It is historical and reads the older merged `data/processed/<author>/corpus.txt` files. Its random chunk split is not used for reported authorship results.

The tracked catalog is `data/books.json`. It records the intended source title and author for each Project Gutenberg ID. Shakespearean criticism by A. C. Bradley is explicitly excluded from Shakespeare's corpus; the original raw file is left untouched. Raw and processed corpus files remain Git-ignored.

## Findings

### Authorship baseline (initial)

- **Data:** 25 books across five authors. Five-fold evaluation holds out one entire book per author in each fold; each book is tested once.
- **Model:** word uni/bi-gram and character 3–5-gram TF-IDF features with a class-balanced LinearSVC.
- **Chunk evaluation:** 625 held-out chunks (25 per book, 500 whitespace-delimited tokens each) achieved **95.7% accuracy** and **95.7% macro-F1**. Fold accuracy ranged from 89.6% to 99.2%.
- **Book evaluation:** aggregating the 25 chunk scores for each held-out book achieved **25/25 correct** (100% accuracy and macro-F1). This is a small result set: five books per author, not a guarantee for arbitrary user-submitted text.
- **Observed errors:** chunk confusions are mainly between Dickens and Austen, and between Christie and Doyle. The Shakespeare data consists of plays while the other authors' samples are novels or short-story collections, so genre may make Shakespeare easier to distinguish.
- **Artifacts:** see `results/authorship_baseline/summary.json` and the adjacent prediction and confusion-matrix CSV files.

This is an initial baseline, not a final product claim. It has no calibrated confidence or unknown-author rejection yet, and the generation model has not been re-evaluated with this book-level methodology.

## License
