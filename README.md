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
5.  **Compare Authorship Models:**
    ```bash
    python src/compare_authorship_models.py
    ```
    This compares word-only, character-only, and combined TF-IDF with LinearSVC, LogisticRegression, and MultinomialNB, using the same book folds and sampled chunks. Results are written to `results/authorship_comparison/`.
6.  **Compare Chunk Parameters:**
    ```bash
    python src/tune_chunk_parameters.py
    ```
    This holds combined TF-IDF + LinearSVC fixed and compares chunk sizes/counts on the same book folds. Results are written to `results/chunk_parameter_comparison/`.
7.  **Check Sampling Robustness and Errors:**
    ```bash
    python src/analyze_authorship_robustness.py
    ```
    This reruns the 500-token x 25 and 1,000-token x 10 settings across five deterministic chunk samples, then writes seed, author, confusion, and per-book error summaries to `results/authorship_robustness/`.
8.  **Evaluate on Untouched Books:**
    ```bash
    python src/evaluate_external_books.py
    ```
    This trains on the original 25 cross-validation books and evaluates on ten separately catalogued books (two per author) that are excluded from the folds and training set. Results are written to `results/external_validation/`.
9.  **Review Generation Experiments:** The available notebook is `notebooks/NLP_MP_Gem.ipynb`. It is historical and reads the older merged `data/processed/<author>/corpus.txt` files. Its random chunk split is not used for reported authorship results.
10. **Historical generation baseline (GPT-2, pipeline pilot):** on a GPU-enabled environment, run:
    ```bash
    python src/train_and_evaluate_generation.py --fold 1
    ```
    This is the historical GPT-2 full-fine-tuning pilot: four training books and one held-out book per author for fold 1. It produces a within-GPT-2 cross-author likelihood table. Checkpoints are saved under the ignored `models/` directory; metrics go to `results/generation_baseline/`. This downloads the base `gpt2` checkpoint if it is not cached. Run it to validate and time the pipeline; its single-fold scores are exploratory, not final resume results.
11. **Prepare an isolated modern-generation environment and smoke-test Qwen3.5:** use a Colab GPU runtime or a separate Python 3.10+ environment. Install the suitable PyTorch build for the runtime, then:
    ```bash
    pip install -r requirements-generation.txt
    python src/smoke_test_modern_generation.py --device cuda
    ```
    This downloads Qwen3.5-0.8B-Base, checks a text-only forward pass, performs one synthetic LoRA optimizer step, then checks adapter save/reload and generation. It writes version/device metadata to the ignored `results/generation_baseline/qwen35_smoke_summary.json`. It does not use project books or produce evaluation metrics. Keep this environment separate from `requirements.txt`, whose older pins support the existing authorship pipeline. See [`docs/generation_experiment_plan.md`](docs/generation_experiment_plan.md) before running full experiments.
    If Colab reports an old preinstalled `torchao` (for example 0.10.0), this smoke test does not need it. Run `!pip uninstall -y torchao`, restart the runtime, return to the cloned project folder, and rerun the smoke test. The script checks for this conflict before downloading model weights. The `causal_conv1d` and `flash-linear-attention` fallback messages are performance warnings; the model can use its reference PyTorch implementations for this smoke test.
12. **Prepare a blind generation-quality review** after the required model arms exist:
    ```bash
    python src/generate_style_eval_samples.py --fold 1
    ```
    The currently implemented generator supports the GPT-2 checkpoints above and produces two samples per shared prompt per model (60 samples total by default). It writes an anonymous rating packet, private key, and ratings template under `results/generation_baseline/`. The modern Qwen prompt/LoRA arms are a planned next implementation and cannot yet be generated by this script. Follow [`docs/generation_human_eval.md`](docs/generation_human_eval.md) to collect ratings. Summarize completed ratings with:
    ```bash
    python src/summarize_generation_ratings.py --ratings results/generation_baseline/human_ratings_completed.csv
    ```
    Generated passages, the key, and ratings are Git-ignored; only aggregate results should be considered for public reporting after review.

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

### Controlled feature and classifier comparison

All nine configurations below use the same five book-held-out folds, 500-token chunks, and 25 sampled chunks per book. This controls the data split and sampling for the comparison.

| Features | Classifier | Chunk accuracy | Chunk macro-F1 | Book accuracy |
|---|---|---:|---:|---:|
| Word + character TF-IDF | LinearSVC | 95.68% | 95.67% | 100% |
| Word TF-IDF | LinearSVC | 95.36% | 95.35% | 100% |
| Word + character TF-IDF | LogisticRegression | 94.24% | 94.20% | 100% |
| Word TF-IDF | LogisticRegression | 93.92% | 93.89% | 100% |
| Character TF-IDF | LinearSVC | 93.12% | 93.05% | 100% |
| Word TF-IDF | MultinomialNB | 92.00% | 91.88% | 100% |
| Character TF-IDF | LogisticRegression | 91.20% | 91.02% | 100% |
| Word + character TF-IDF | MultinomialNB | 81.28% | 80.07% | 92% |
| Character TF-IDF | MultinomialNB | 68.64% | 63.61% | 68% |

Combined word+character features with LinearSVC lead among these configurations, but only by 0.32 percentage points of chunk accuracy over word-only LinearSVC. This is a provisional comparison using default settings, one fixed chunk sample, and only 25 books. It does not establish a globally best classifier or optimal chunk size/count. Book accuracy also saturates at 100% for most configurations, so chunk metrics and per-author errors are more informative here. Results and per-fold breakdowns are in `results/authorship_comparison/`.

### Chunk size and sample-count comparison

The leading combined-feature LinearSVC was evaluated with 250-, 500-, and 1,000-token chunks. At matched sampled-token budgets, longer chunks had higher passage-level macro-F1:

| Sampled tokens per book | 250-token chunks | 500-token chunks | 1,000-token chunks |
|---:|---:|---:|---:|
| 2,500 | 10 chunks: 86.11% | 5 chunks: 89.52% | — |
| 5,000 | 20 chunks: 87.56% | 10 chunks: 91.47% | 5 chunks: 95.88% |
| 10,000 | 40 chunks: 92.54% | 20 chunks: 93.71% | 10 chunks: 97.59% |
| 17,000 | — | 34 chunks: 95.39% | 17 chunks: 98.83% |

At 500 tokens per chunk, increasing the sampled count from 5 to 34 raised pooled chunk macro-F1 from 89.52% to 95.39%. Book accuracy stayed at 100% in all 11 settings, so this dataset’s book-level metric is saturated and cannot select a useful chunk count. Longer passages contain more evidence and are an easier classification task; these results do not establish that 1,000 tokens is best for every upload use case. The parameter sweep also reuses the same 25 books/folds for selection, so its scores are exploratory rather than an independent final estimate. Full aggregate, fold, and book predictions are in `results/chunk_parameter_comparison/`.

### Sampling robustness and error analysis

The 500 x 25 and 1,000 x 10 settings were each evaluated over five deterministic chunk samples, with the same held-out books and LinearSVC setup. Seed `20261106` reproduces the original 500 x 25 sample.

| Setting | Mean chunk accuracy (std. dev.) | Mean chunk macro-F1 (std. dev.) | Accuracy range across seeds | Book accuracy in each seed |
|---|---:|---:|---:|---:|
| 500 tokens x 25 | 95.33% (0.48 pp) | 95.29% (0.49 pp) | 94.40–95.68% | 100% |
| 1,000 tokens x 10 | 96.56% (0.41 pp) | 96.54% (0.41 pp) | 96.00–97.20% | 100% |

The passage results are stable across these five samples. All 25 whole-book predictions remained correct for every seed and setting, but this small book-level test remains saturated. The largest recurring chunk confusions were Dickens as Austen (43 errors at 500 tokens; 14 at 1,000) and Doyle as Christie (27 and 7, respectively). The books with the most chunk errors were *Bleak House* (39/125 tested chunks at 500 tokens; 14/50 at 1,000), *The Valley of Fear* (24/125; 7/50), and *Oliver Twist* (18/125; 7/50). A review of representative error passages found ordinary narrative/dialogue text, with no obvious wrapper or label contamination in those examples.

These five samples reuse the same 25 books and book folds; they measure sensitivity to which chunks are sampled, not variation across new books or authors. Full seed-level metrics, chunk predictions, author scores, confusion matrices, and per-book error counts are in `results/authorship_robustness/`.

### Generation evaluation status

The historical notebook fine-tunes a separate GPT-2 model per author, but splits randomly sampled token blocks from each merged author corpus into train and validation. Blocks from the same book can therefore appear on both sides. It logs validation loss but does not produce a consistent book-held-out cross-author perplexity comparison. Its generation review uses a fixed prompt and a few samples, with manually noted coherence issues and basic stylometric comparisons; there are no saved checkpoints in this repository for a reproducible rerun.

`src/train_and_evaluate_generation.py` sets up the next controlled baseline using the per-book corpus and existing fold 1. It uses 512-token blocks that never cross book boundaries, trains each author model on four books, and evaluates on the held-out book. The cross-author table also scores base GPT-2 on the same held-out books. This experiment has not been run in this environment: no model checkpoints are present and neither CUDA nor Apple MPS is available. Run it on a GPU-enabled environment; these results remain preliminary until the training corpus is expanded and a fresh external evaluation is reserved.

The revised generation experiment is specified in [`docs/generation_experiment_plan.md`](docs/generation_experiment_plan.md). GPT-2 remains the historical reference. The primary modern comparison is a frozen Qwen3.5-0.8B-Base model given author examples versus the same base model adapted with an author-specific LoRA adapter. This isolates the value of adaptation more clearly than comparing unrelated model checkpoints. The model's hybrid multimodal architecture and newer library requirements add runtime risk, so a separate environment and smoke test are prepared before full training. No generation metrics have been produced yet.

### External validation on new books

The classifier was trained on the original 25 books and evaluated on ten additional books held out completely from training and cross-validation: two books per author. The catalog labels them `evaluation_group: external_validation`; `src/build_book_folds.py` excludes that group. The external evaluator checks that training and test IDs do not overlap.

| Model input | Training chunks | External chunks | Chunk accuracy | Chunk macro-F1 | Whole books correct |
|---|---:|---:|---:|---:|---:|
| 500 tokens x 25 per book | 625 | 250 | 98.80% | 98.80% | 10/10 |
| 1,000 tokens x 10 per book | 250 | 100 | 99.00% | 99.00% | 10/10 |

At 500 tokens, three external chunks were misclassified: two from *The Mysterious Affair at Styles* and one from *Great Expectations*. At 1,000 tokens, one chunk from *Great Expectations* was misclassified. Combining chunk scores gave the correct author for all ten books in both settings. This is encouraging evidence of transfer to new works, but the external set is still small (two books per author), and Shakespeare is represented by plays while most other authors are represented by prose fiction. Treat these as an external check, not a universal accuracy guarantee. Detailed outputs are in `results/external_validation/`.

The representative error passages involve courtroom/family discussion, detective reasoning, and dialogue about social class. They broadly match the internal Dickens/Austen and Christie/Doyle confusion patterns, though this qualitative review does not prove why the classifier erred. See `results/external_validation/error_review.md`; keep this holdout out of training and avoid using it repeatedly to tune choices.

## License
