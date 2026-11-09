"""Generate matched, blinded samples for human evaluation of author style."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[1]
FOLDS_PATH = ROOT / "data" / "evaluation" / "book_folds.csv"
MODELS_DIR = ROOT / "models" / "gpt2_by_author"
OUT_DIR = ROOT / "results" / "generation_baseline"
SEED = 20261106
PROMPTS = [
    ("arrival", "A traveller arrives at an unfamiliar town on a rainy evening."),
    ("letter", "A letter arrives at a country house, bringing unexpected news."),
    ("dinner", "At dinner, a guest makes a remark that changes the mood."),
    ("reunion", "Two friends meet after a long separation and discuss an old promise."),
    ("interruption", "A quiet evening is interrupted by a knock at the door."),
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("Apple MPS was requested but is not available.")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt < 1 or args.max_new_tokens < 1:
        raise ValueError("Sample count and max-new-tokens must be positive.")
    with FOLDS_PATH.open(encoding="utf-8", newline="") as stream:
        fold_rows = list(csv.DictReader(stream))
    authors = sorted({row["author_key"] for row in fold_rows})
    fold_ids = {int(row["held_out_fold"]) for row in fold_rows}
    if args.fold not in fold_ids:
        raise ValueError(f"Unknown fold {args.fold}; available folds: {sorted(fold_ids)}")

    device = choose_device(args.device)
    if device.type == "cpu":
        print("Warning: generation on CPU may take several minutes.")
    model_specs = [(author, MODELS_DIR / f"fold_{args.fold}" / author) for author in authors]
    model_specs.append(("base_gpt2", args.base_model))
    for name, path in model_specs[:-1]:
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(
                f"Missing fine-tuned checkpoint for {name}: {path}. "
                f"Run src/train_and_evaluate_generation.py --fold {args.fold} first."
            )

    rows = []
    for model_index, (model_author, model_path) in enumerate(model_specs):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()
        for prompt_index, (prompt_id, prompt) in enumerate(PROMPTS):
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            for sample_index in range(args.samples_per_prompt):
                sample_seed = SEED + model_index * 10_000 + prompt_index * 100 + sample_index
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
                        pad_token_id=tokenizer.eos_token_id,
                    )
                continuation = tokenizer.decode(
                    generated[0, encoded["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                rows.append({
                    "model_author": model_author,
                    "fold": args.fold,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "sample_index": sample_index + 1,
                    "seed": sample_seed,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0.85,
                    "top_k": 50,
                    "top_p": 0.95,
                    "repetition_penalty": 1.1,
                    "continuation": continuation,
                })
                print(f"Generated {model_author}/{prompt_id}/{sample_index + 1}")
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_fields = list(rows[0])
    write_csv(OUT_DIR / "generated_samples.csv", rows, sample_fields)

    blind_rows = []
    key_rows = []
    shuffled_rows = rows.copy()
    random.Random(SEED).shuffle(shuffled_rows)
    for index, row in enumerate(shuffled_rows, start=1):
        sample_id = f"sample_{index:04d}"
        blind_rows.append({
            "sample_id": sample_id,
            "prompt_id": row["prompt_id"],
            "prompt": row["prompt"],
            "continuation": row["continuation"],
        })
        key_rows.append({"sample_id": sample_id, "model_author": row["model_author"]})
    write_csv(OUT_DIR / "human_eval_packet.csv", blind_rows,
              ["sample_id", "prompt_id", "prompt", "continuation"])
    write_csv(OUT_DIR / "human_eval_key.csv", key_rows, ["sample_id", "model_author"])
    rating_fields = ["sample_id", "reviewer_id", "fluency_1_to_5",
                     "coherence_1_to_5", "topic_fit_1_to_5", "attribution_guess",
                     "confidence_1_to_5", "notes"]
    write_csv(OUT_DIR / "human_ratings_template.csv", [], rating_fields)
    print(f"Wrote {len(rows)} samples and blinded packet to {OUT_DIR}.")


if __name__ == "__main__":
    main()
