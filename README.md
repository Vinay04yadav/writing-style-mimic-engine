# Writing Style Mimic Engine

## Overview

This project aims to develop a system that analyzes an author's writing style (vocabulary, syntax, sentence structure, tone) using NLP techniques and generates text mimicking that style. This was developed as a mini-project for [Your Course/Purpose].

## Features

- Downloads public domain texts from Project Gutenberg for specified authors.
- Preprocesses text data (sentence splitting, tokenization, lowercasing).
- Performs stylometric analysis (sentence length, TTR, POS tags, punctuation, readability).
- Fine-tunes GPT-2 models on individual author corpora.
- Generates text mimicking learned author styles.
- Includes notebooks for analysis and generation.

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
4.  **Download NLTK data:**
    ```bash
    python -m nltk.downloader punkt averaged_perceptron_tagger
    ```

## Usage

1.  **Download Data:** Modify the `BOOKS_TO_DOWNLOAD` list in `src/download_gutenberg.py` with desired authors/book IDs from Project Gutenberg. Then run:
    ```bash
    python src/download_gutenberg.py
    ```
    _(This populates `data/raw`)_
2.  **Preprocess Data:** Run the preprocessing script:
    ```bash
    python src/preprocess_corpus.py
    ```
    _(This populates `data/processed`)_
3.  **Analyze Style:** Open and run the cells in `notebooks/01_style_analysis.ipynb`.
4.  **Train Models:** Open and run the cells in `notebooks/02_training_generation.ipynb` (or your training notebook). **Note:** Training requires significant time and a GPU (Google Colab recommended). Ensure paths to data/models (likely on Google Drive) are set correctly within the notebook.
5.  **Generate Text:** Use the latter part of `notebooks/02_training_generation.ipynb` (or a separate generation notebook) to load saved models and generate text.

## Findings

_(Briefly summarize key stylistic differences found or link to the analysis notebook)_

## License
