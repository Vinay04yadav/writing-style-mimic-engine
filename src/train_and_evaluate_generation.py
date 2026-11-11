"""Train per-author GPT-2 models and score them on book-held-out text.

The selected fold is held out by complete book. Each author model trains only
on that author's other books. After training, every model (and base GPT-2) is
scored on each author's held-out book, producing a cross-author perplexity
matrix. Model files are written under the ignored ``models/`` directory.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOLDS_PATH = PROJECT_ROOT / "data" / "evaluation" / "book_folds.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "gpt2_by_author"
RESULTS_DIR = PROJECT_ROOT / "results" / "generation_baseline"
DEFAULT_MODEL = "gpt2"
SEED = 20261106


class CausalBookBlocks(Dataset):
    """Non-overlapping fixed-length token blocks that never cross book edges."""

    def __init__(self, rows: list[dict], tokenizer, block_size: int):
        self.examples = []
        self.books = []
        self.token_count = 0
        self.blocks_by_book = {}

        for row in rows:
            path = PROJECT_ROOT / row["processed_path"]
            text = path.read_text(encoding="utf-8", errors="replace")
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids.append(tokenizer.eos_token_id)
            usable_length = len(token_ids) // block_size * block_size
            book_blocks = [
                token_ids[start:start + block_size]
                for start in range(0, usable_length, block_size)
            ]
            if not book_blocks:
                raise ValueError(f"No full {block_size}-token blocks in {path}.")
            self.examples.extend(book_blocks)
            self.books.append(row)
            self.blocks_by_book[int(row["gutenberg_id"])] = len(book_blocks)
            self.token_count += sum(len(block) - 1 for block in book_blocks)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.examples[index], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def read_assignments() -> tuple[list[dict], list[str], list[int]]:
    with FOLDS_PATH.open(encoding="utf-8", newline="") as folds_file:
        rows = list(csv.DictReader(folds_file))
    authors = sorted({row["author_key"] for row in rows})
    fold_ids = sorted({int(row["held_out_fold"]) for row in rows})
    return rows, authors, fold_ids


def training_arguments(args, output_dir: Path) -> TrainingArguments:
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    values = dict(
        output_dir=str(output_dir),
        save_strategy="no",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=False,
        logging_steps=100,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    values[strategy_name] = "no"
    return TrainingArguments(**values)


def make_trainer(model, dataset, args, output_dir: Path, *, evaluate: bool):
    return Trainer(
        model=model,
        args=training_arguments(args, output_dir),
        data_collator=default_data_collator,
        train_dataset=dataset if not evaluate else None,
        eval_dataset=dataset if evaluate else None,
    )


def score_perplexity(model, tokenizer, dataset, args, output_dir: Path) -> float:
    trainer = make_trainer(
        model, dataset, args, output_dir, evaluate=True
    )
    loss = float(trainer.evaluate()["eval_loss"])
    del trainer
    return math.exp(loss)


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def train_author_model(author, fold_id, train_rows, validation_row, tokenizer, args):
    model_path = MODELS_DIR / f"fold_{fold_id}" / author
    if model_path.exists() and any(model_path.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Model directory already exists: {model_path}. "
            "Choose another fold/output path or pass --overwrite."
        )
    if args.overwrite and model_path.exists():
        import shutil
        shutil.rmtree(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    train_dataset = CausalBookBlocks(train_rows, tokenizer, args.block_size)
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_arguments(args, model_path),
        data_collator=default_data_collator,
        train_dataset=train_dataset,
    )

    fixed_schedule_note = (
        "The held-out book is never used for epoch/checkpoint selection. "
        "Train for the configured fixed epoch count and evaluate only afterward."
    )
    print(
        f"Training {author}: {len(train_rows)} books, {len(train_dataset):,} blocks; "
        f"held-out evaluation={validation_row['title']} after {args.epochs} fixed epochs."
    )
    trainer.train()
    model.config.use_cache = True
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(model_path)
    del trainer
    validation_dataset = CausalBookBlocks([validation_row], tokenizer, args.block_size)
    final_perplexity = score_perplexity(
        model, tokenizer, validation_dataset, args, model_path / "heldout_eval"
    )
    metadata = {
        "author": author,
        "fold": fold_id,
        "model_checkpoint": args.model_checkpoint,
        "block_size": args.block_size,
        "seed": SEED,
        "training_books": [
            {"gutenberg_id": int(row["gutenberg_id"]), "title": row["title"]}
            for row in train_rows
        ],
        "heldout_evaluation_book": {
            "gutenberg_id": int(validation_row["gutenberg_id"]),
            "title": validation_row["title"],
        },
        "training_blocks": len(train_dataset),
        "heldout_evaluation_blocks": len(validation_dataset),
        "heldout_evaluation_tokens_scored": validation_dataset.token_count,
        "checkpoint_selection": "fixed_epochs_no_heldout_selection",
        "evaluation_note": fixed_schedule_note,
        "final_perplexity": final_perplexity,
    }
    (model_path / "training_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    del model, train_dataset, validation_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_path


def evaluate_cross_author(models, heldout, tokenizer, args) -> list[dict]:
    rows = []
    evaluation_root = RESULTS_DIR / "evaluation_runs"
    for model_author, model_path in models.items():
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = True
        for test_author, (test_row, dataset) in heldout.items():
            run_path = evaluation_root / model_author / test_author
            perplexity = score_perplexity(model, tokenizer, dataset, args, run_path)
            rows.append({
                "model_author": model_author,
                "test_author": test_author,
                "test_book_id": int(test_row["gutenberg_id"]),
                "test_book": test_row["title"],
                "blocks": len(dataset),
                "tokens_scored": dataset.token_count,
                "perplexity": perplexity,
            })
            print(f"{model_author} on {test_author}: perplexity={perplexity:.2f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = True
    for test_author, (test_row, dataset) in heldout.items():
        run_path = evaluation_root / "base_gpt2" / test_author
        perplexity = score_perplexity(
            base_model, tokenizer, dataset, args, run_path
        )
        rows.append({
            "model_author": "base_gpt2",
            "test_author": test_author,
            "test_book_id": int(test_row["gutenberg_id"]),
            "test_book": test_row["title"],
            "blocks": len(dataset),
            "tokens_scored": dataset.token_count,
            "perplexity": perplexity,
        })
        print(f"base_gpt2 on {test_author}: perplexity={perplexity:.2f}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--author", help="Train one author; default trains all authors.")
    parser.add_argument("--model-checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    assignments, authors, fold_ids = read_assignments()
    if args.fold not in fold_ids:
        raise ValueError(f"Unknown fold {args.fold}; available folds: {fold_ids}.")
    if args.author and args.author not in authors:
        raise ValueError(f"Unknown author {args.author}; available authors: {authors}.")
    if args.block_size < 2 or args.block_size > 1024:
        raise ValueError("GPT-2 block size must be between 2 and 1024 tokens.")
    selected_authors = [args.author] if args.author else authors

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = min(tokenizer.model_max_length, 1024)
    if args.block_size > tokenizer.model_max_length:
        raise ValueError(
            f"Block size {args.block_size} exceeds tokenizer maximum "
            f"{tokenizer.model_max_length}."
        )

    heldout = {}
    models = {}
    for author in selected_authors:
        author_rows = [row for row in assignments if row["author_key"] == author]
        validation = [row for row in author_rows if int(row["held_out_fold"]) == args.fold]
        train_rows = [row for row in author_rows if int(row["held_out_fold"]) != args.fold]
        if len(validation) != 1 or len(train_rows) != 4:
            raise ValueError(
                f"Fold {args.fold} expected one validation and four training books "
                f"for {author}; got {len(validation)} and {len(train_rows)}."
            )
        validation_dataset = CausalBookBlocks(validation, tokenizer, args.block_size)
        heldout[author] = (validation[0], validation_dataset)
        models[author] = train_author_model(
            author, args.fold, train_rows, validation[0], tokenizer, args
        )

    # A single-author run trains that model but cannot create a full cross-author matrix.
    if len(models) != len(authors):
        print("Trained the selected author model; run without --author for the full matrix.")
        return

    cross_rows = evaluate_cross_author(models, heldout, tokenizer, args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(
        RESULTS_DIR / f"perplexity_fold_{args.fold}.csv",
        cross_rows,
        list(cross_rows[0]),
    )
    config = {
        "fold": args.fold,
        "model_checkpoint": args.model_checkpoint,
        "block_size": args.block_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": SEED,
        "training_books_per_author": 4,
        "heldout_books_per_author": 1,
        "external_validation_books_used": False,
    }
    (RESULTS_DIR / f"config_fold_{args.fold}.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved cross-author perplexities to {RESULTS_DIR}.")


if __name__ == "__main__":
    main()
