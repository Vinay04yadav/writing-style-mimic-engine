"""Train/evaluate Qwen author adapters and create a matched prompt-vs-LoRA packet.

Run from the project root in the isolated requirements-generation environment.
The held-out book is never used to pick epochs or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
FOLDS_PATH = ROOT / "data" / "evaluation" / "book_folds.csv"
MODEL_ROOT = ROOT / "models" / "qwen35_lora"
RESULTS_ROOT = ROOT / "results" / "generation_baseline" / "qwen35"
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"
SEED = 20261106
LORA_RANK = 8
LORA_ALPHA = 16
PROMPTS = [
    ("arrival", "A traveller arrives at an unfamiliar town on a rainy evening."),
    ("letter", "A letter arrives at a country house, bringing unexpected news."),
    ("dinner", "At dinner, a guest makes a remark that changes the mood."),
    ("reunion", "Two friends meet after a long separation and discuss an old promise."),
    ("interruption", "A quiet evening is interrupted by a knock at the door."),
]


class TokenBlocks(Dataset):
    """Fixed-length next-token blocks, built separately inside each source book."""

    def __init__(self, rows: list[dict], tokenizer, block_size: int):
        self.blocks: list[list[int]] = []
        self.book_blocks: dict[int, int] = {}
        self.scored_tokens = 0
        self.decoded_bytes = 0
        for row in rows:
            path = ROOT / row["processed_path"]
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing processed text {path}. In Colab run download_gutenberg.py, "
                    "preprocess_corpus.py, and build_book_folds.py first."
                )
            text = path.read_text(encoding="utf-8", errors="replace")
            ids = tokenizer.encode(text, add_special_tokens=False)
            body_ids = ids.copy()
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            usable = len(ids) // block_size * block_size
            if usable < block_size:
                raise ValueError(f"No complete {block_size}-token block in {path}.")
            book_blocks = [ids[i:i + block_size] for i in range(0, usable, block_size)]
            self.blocks.extend(book_blocks)
            self.book_blocks[int(row["gutenberg_id"])] = len(book_blocks)
            self.scored_tokens += sum(len(block) - 1 for block in book_blocks)
            body_limit = min(len(body_ids), usable)
            decoded = tokenizer.decode(
                body_ids[:body_limit], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self.decoded_bytes += len(decoded.encode("utf-8"))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        ids = torch.tensor(self.blocks[index], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


def read_folds():
    if not FOLDS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {FOLDS_PATH}. Download and preprocess the catalogued books, "
            "then run src/build_book_folds.py."
        )
    with FOLDS_PATH.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    authors = sorted({row["author_key"] for row in rows})
    folds = sorted({int(row["held_out_fold"]) for row in rows})
    return rows, authors, folds


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_torchao_compatible():
    import importlib.metadata
    from packaging.version import Version

    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return
    if Version(version) < Version("0.16.0"):
        raise RuntimeError(
            f"Found torchao {version}; PEFT LoRA dispatch needs torchao >=0.16 if it is "
            "installed. This experiment does not use torchao quantization. In Colab run "
            "`!pip uninstall -y torchao`, restart the runtime, then retry."
        )


def fixed_training_args(args, author_output: Path) -> TrainingArguments:
    # Transformers 5.22 uses eval_strategy. The held-out book is deliberately
    # not passed to Trainer: it is scored only after the fixed training schedule.
    return TrainingArguments(
        output_dir=str(author_output / "trainer_state"),
        eval_strategy="no",
        save_strategy="no",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        # Transformers v5.2 removed warmup_ratio; a float below 1 is the ratio.
        warmup_steps=0.03,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_only_model=True,
    )


def author_rows(assignments, author, fold):
    rows = [row for row in assignments if row["author_key"] == author]
    heldout = [row for row in rows if int(row["held_out_fold"]) == fold]
    train = [row for row in rows if int(row["held_out_fold"]) != fold]
    if len(heldout) != 1 or len(train) != 4:
        raise ValueError(
            f"Fold {fold} expected 4 train books and 1 held-out book for {author}; "
            f"found {len(train)} and {len(heldout)}."
        )
    return train, heldout[0]


def create_adapter(author, fold, train_rows, tokenizer, snapshot, args):
    output_dir = MODEL_ROOT / f"fold_{fold}" / author
    summary_path = output_dir / "training_summary.json"
    expected = {
        "author": author,
        "fold": fold,
        "base_model": args.model,
        "base_revision": args.resolved_revision,
        "block_size": args.block_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "seed": SEED,
    }
    if summary_path.is_file() and not args.overwrite:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in expected.items()):
            raise FileExistsError(
                f"Existing adapter {output_dir} has a different run configuration. "
                "Use --overwrite to replace it."
            )
        print(f"Reusing matching adapter: {author}")
        return output_dir
    if output_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(output_dir)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Partial model directory exists: {output_dir}. Use --overwrite or remove it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TokenBlocks(train_rows, tokenizer, args.block_size)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.float32,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(
        f"Training {author}: {len(train_rows)} books, {len(dataset):,} blocks, "
        f"{trainable_count:,}/{total_count:,} trainable parameters, "
        f"{args.epochs} fixed epoch(s)."
    )
    trainer = Trainer(
        model=model,
        args=fixed_training_args(args, output_dir),
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    train_result = trainer.train()
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    metadata = {
        **expected,
        "train_books": [
            {"gutenberg_id": int(row["gutenberg_id"]), "title": row["title"]}
            for row in train_rows
        ],
        "training_blocks": len(dataset),
        "training_scored_tokens": dataset.scored_tokens,
        "training_loss": float(train_result.training_loss),
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "selection": "fixed_epoch_schedule; held-out book not used during training",
        "torch": torch.__version__,
    }
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del trainer, model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


def score_model(model, dataset, batch_size: int, device) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    nll_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = ids[:, 1:].contiguous()
            nll_sum += float(F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1), reduction="sum",
            ).item())
            token_count += shifted_labels.numel()
    mean_nll = nll_sum / token_count
    return {
        "blocks": len(dataset),
        "tokens_scored": token_count,
        "decoded_utf8_bytes": dataset.decoded_bytes,
        "mean_nll_nats_per_token": mean_nll,
        "perplexity_within_qwen_tokenizer": math.exp(min(mean_nll, 80)),
        "bits_per_byte_supplementary": nll_sum / math.log(2) / dataset.decoded_bytes,
    }


def evaluate_matrix(assignments, authors, fold, models, tokenizer, snapshot, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heldout = {}
    for author in authors:
        _, row = author_rows(assignments, author, fold)
        heldout[author] = (row, TokenBlocks([row], tokenizer, args.block_size))

    rows = []
    base = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32).to(device)
    base.config.use_cache = True
    for test_author, (row, dataset) in heldout.items():
        scores = score_model(base, dataset, args.eval_batch_size, device)
        rows.append({
            "arm": "qwen_frozen_base",
            "model_author": "base",
            "test_author": test_author,
            "test_book_id": int(row["gutenberg_id"]),
            "test_book": row["title"],
            **scores,
        })
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for model_author, adapter_path in models.items():
        base = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32).to(device)
        model = PeftModel.from_pretrained(base, adapter_path).eval()
        model.config.use_cache = True
        for test_author, (row, dataset) in heldout.items():
            scores = score_model(model, dataset, args.eval_batch_size, device)
            rows.append({
                "arm": "qwen_author_lora",
                "model_author": model_author,
                "test_author": test_author,
                "test_book_id": int(row["gutenberg_id"]),
                "test_book": row["title"],
                **scores,
            })
            print(
                f"LoRA {model_author} on {test_author}: "
                f"PPL={scores['perplexity_within_qwen_tokenizer']:.2f}"
            )
        del model, base
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = RESULTS_ROOT / f"fold_{fold}"
    write_csv(out_dir / "cross_author_likelihood.csv", rows)
    return heldout


def sample_excerpt(row, tokenizer, seed: int, token_count: int = 160) -> str:
    text = (ROOT / row["processed_path"]).read_text(encoding="utf-8", errors="replace")
    ids = tokenizer.encode(text, add_special_tokens=False)
    max_start = max(0, len(ids) - token_count)
    if max_start == 0:
        start = 0
    else:
        low = min(len(ids) // 4, max_start)
        high = min(max_start, max(low, (3 * len(ids)) // 4 - token_count))
        start = random.Random(seed).randint(low, high)
    return tokenizer.decode(ids[start:start + token_count], skip_special_tokens=True).strip()


def chat_inputs(tokenizer, content: str, device):
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded.to(device)
    return tokenizer(content, return_tensors="pt").to(device)


def generate_outputs(args, assignments, authors, tokenizer, snapshot, models):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32).to(device).eval()
    all_rows = []
    for author_index, author in enumerate(authors):
        train_rows, _ = author_rows(assignments, author, args.fold)
        examples = [
            sample_excerpt(row, tokenizer, SEED + author_index * 100 + i, token_count=128)
            for i, row in enumerate(train_rows)
        ]
        for arm in ("qwen_fewshot", "qwen_author_lora"):
            model = seed_model
            base = None
            adapter_model = None
            if arm == "qwen_author_lora":
                base = AutoModelForCausalLM.from_pretrained(
                    snapshot, dtype=torch.float32
                ).to(device)
                adapter_model = PeftModel.from_pretrained(
                    base, models[author]
                ).eval()
                adapter_model.config.use_cache = True
                model = adapter_model
            for prompt_index, (prompt_id, topic) in enumerate(PROMPTS):
                if arm == "qwen_fewshot":
                    prompt_text = (
                        "Write an original literary passage inspired by the writing examples below. "
                        "Do not copy their sentences or reuse plot details. Return only the new passage.\n\n"
                        + "\n\n".join(
                            f"Example {example_index + 1}:\n{example}"
                            for example_index, example in enumerate(examples)
                        )
                        + "\n\n"
                        f"New topic: {topic}"
                    )
                else:
                    prompt_text = (
                        "Write an original literary passage. Return only the passage.\n\n"
                        f"New topic: {topic}"
                    )
                encoded = chat_inputs(tokenizer, prompt_text, device)
                for sample_index in range(args.samples_per_prompt):
                    sample_seed = SEED + author_index * 10_000 + prompt_index * 100 + sample_index
                    set_seed(sample_seed)
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            do_sample=True,
                            max_new_tokens=args.max_new_tokens,
                            temperature=0.85,
                            top_k=50,
                            top_p=0.95,
                            repetition_penalty=1.1,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    continuation = tokenizer.decode(
                        generated[0, encoded["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ).strip()
                    all_rows.append({
                        "arm": arm,
                        "target_author": author,
                        "fold": args.fold,
                        "prompt_id": prompt_id,
                        "topic": topic,
                        "sample_index": sample_index + 1,
                        "seed": sample_seed,
                        "base_model": args.model,
                        "base_revision": args.resolved_revision,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": 0.85,
                        "top_k": 50,
                        "top_p": 0.95,
                        "repetition_penalty": 1.1,
                        "continuation": continuation,
                    })
                    print(f"Generated {arm}/{author}/{prompt_id}/{sample_index + 1}")
            if adapter_model is not None:
                del adapter_model, base
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    del seed_model
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_ROOT / f"generated_samples_fold_{args.fold}.csv", all_rows)

    # Balanced first-round human subset: two prompts per author per Qwen arm.
    sample_lookup = {}
    selected = []
    for row in all_rows:
        key = (row["arm"], row["target_author"])
        if key not in sample_lookup:
            sample_lookup[key] = 0
        if row["sample_index"] == 1 and sample_lookup[key] < 2:
            selected.append(row)
            sample_lookup[key] += 1
    random.Random(SEED).shuffle(selected)
    packet, key_rows = [], []
    for index, row in enumerate(selected, start=1):
        sample_id = f"sample_{index:04d}"
        packet.append({
            "sample_id": sample_id,
            "prompt_id": row["prompt_id"],
            "topic": row["topic"],
            "continuation": row["continuation"],
        })
        key_rows.append({
            "sample_id": sample_id,
            "arm": row["arm"],
            "target_author": row["target_author"],
        })
    packet_dir = RESULTS_ROOT / f"fold_{args.fold}"
    write_csv(packet_dir / "human_eval_packet.csv", packet)
    write_csv(packet_dir / "human_eval_key.csv", key_rows)
    write_csv(packet_dir / "human_ratings_template.csv", [], [
        "sample_id", "reviewer_id", "fluency_1_to_5", "coherence_1_to_5",
        "topic_fit_1_to_5", "attribution_guess", "confidence_1_to_5", "notes",
    ])
    print(f"Wrote {len(all_rows)} generated samples and {len(packet)} blind rating samples.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "generate", "all"), default="all")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--author", help="Train one author adapter; default trains all five.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Use a model commit SHA to reproduce a previous run.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    global LoraConfig, PeftModel, get_peft_model
    args = parse_args()
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "Qwen training needs the isolated generation dependencies. "
            "Install requirements-generation.txt in the GPU environment."
        ) from exc
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 and --bf16.")
    if args.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16; use fp32 or --fp16.")
    if args.block_size < 2 or args.block_size > 4096:
        raise ValueError("block-size must be between 2 and 4096 tokens.")
    if args.samples_per_prompt < 1 or args.max_new_tokens < 1:
        raise ValueError("Sample count and max-new-tokens must be positive.")
    if args.mode in ("train", "all") and not torch.cuda.is_available():
        raise RuntimeError("Qwen training requires a CUDA GPU; choose a GPU runtime such as Colab.")
    ensure_torchao_compatible()
    assignments, authors, folds = read_folds()
    if args.fold not in folds:
        raise ValueError(f"Unknown fold {args.fold}; available: {folds}")
    if args.author and args.author not in authors:
        raise ValueError(f"Unknown author {args.author}; available: {authors}")
    if args.author and args.mode != "train":
        raise ValueError("--author is only supported with --mode train.")
    if args.mode == "generate":
        config_path = RESULTS_ROOT / f"fold_{args.fold}" / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing {config_path}; train all five adapters first with --mode train."
            )
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.model != DEFAULT_MODEL and args.model != saved_config["model"]:
            raise ValueError("--model differs from the trained adapters' base model.")
        if args.revision and args.revision != saved_config["resolved_revision"]:
            raise ValueError("--revision differs from the trained adapters' base revision.")
        args.model = saved_config["model"]
        args.revision = saved_config["resolved_revision"]
    elif args.revision is None:
        args.revision = "main"
    set_seed(SEED)
    snapshot = Path(snapshot_download(repo_id=args.model, revision=args.revision))
    args.resolved_revision = snapshot.name
    print(
        f"model={args.model} resolved_revision={args.resolved_revision} "
        f"torch={torch.__version__} device={'cuda' if torch.cuda.is_available() else 'cpu'}"
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    if tokenizer.eos_token_id is None:
        raise ValueError("Qwen tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = min(tokenizer.model_max_length, 4096)
    if args.block_size > tokenizer.model_max_length:
        raise ValueError(f"block-size exceeds tokenizer max {tokenizer.model_max_length}.")

    selected = [args.author] if args.author else authors
    models = {}
    if args.mode in ("train", "all"):
        for author in selected:
            train_rows, heldout = author_rows(assignments, author, args.fold)
            models[author] = create_adapter(
                author, args.fold, train_rows, tokenizer, snapshot, args
            )
        if len(models) == len(authors):
            evaluate_matrix(assignments, authors, args.fold, models,
                            tokenizer, snapshot, args)
        else:
            print("Single-author adapter trained; run all authors to create the matrix.")
    if args.mode in ("generate", "all"):
        models = {
            author: MODEL_ROOT / f"fold_{args.fold}" / author
            for author in authors
        }
        missing = [author for author, path in models.items()
                   if not (path / "adapter_config.json").is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing trained Qwen adapters for {missing}; run with --mode train first."
            )
        generate_outputs(args, assignments, authors, tokenizer, snapshot, models)
    if args.mode in ("train", "all") and len(selected) == len(authors):
        config = {
            "fold": args.fold,
            "model": args.model,
            "resolved_revision": args.resolved_revision,
            "block_size": args.block_size,
            "epochs_fixed": args.epochs,
            "learning_rate": args.learning_rate,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "precision": "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
            "seed": SEED,
            "heldout_used_for_selection": False,
            "external_books_used": False,
        }
        config_path = RESULTS_ROOT / f"fold_{args.fold}" / "run_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
