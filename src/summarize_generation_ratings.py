"""Summarize human ratings against the locally stored blind-evaluation key."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results" / "generation_baseline"
SCORE_COLUMNS = ("fluency_1_to_5", "coherence_1_to_5", "topic_fit_1_to_5")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--ratings", type=Path, help="Completed ratings CSV; defaults to human_ratings_template.csv")
    args = parser.parse_args()
    directory = args.directory
    ratings_path = args.ratings or directory / "human_ratings_template.csv"
    key_path = directory / "human_eval_key.csv"
    ratings, key_rows = read_csv(ratings_path), read_csv(key_path)
    if not ratings:
        raise ValueError(f"No ratings found in {ratings_path}; fill a copy of the template first.")
    key = {row["sample_id"]: row for row in key_rows}
    allowed_guesses = {
        row.get("target_author", "") or row.get("model_author", "")
        for row in key.values()
    } - {"base_gpt2", ""}
    allowed_guesses.add("unsure")
    grouped = defaultdict(list)
    for row in ratings:
        sample_id = row["sample_id"].strip()
        if sample_id not in key:
            raise ValueError(f"Unknown sample_id {sample_id!r} in {ratings_path}.")
        for column in SCORE_COLUMNS + ("confidence_1_to_5",):
            if row.get(column, "").strip():
                score = int(row[column])
                if score < 1 or score > 5:
                    raise ValueError(f"{column} must be between 1 and 5 (sample {sample_id}).")
        identity = key[sample_id]
        arm = identity.get("arm") or identity.get("model_author", "unknown")
        author = identity.get("target_author") or identity.get("model_author", "")
        guess = row.get("attribution_guess", "").strip()
        if guess and guess not in allowed_guesses:
            raise ValueError(
                f"Unknown attribution_guess {guess!r}; use one of "
                f"{sorted(allowed_guesses)}."
            )
        row["_arm"] = arm
        row["_target_author"] = author
        grouped[(arm, author)].append(row)

    result = []
    for arm, author in sorted(grouped):
        group = grouped[(arm, author)]
        out = {"arm": arm, "target_author": author, "ratings": len(group),
               "unique_samples_rated": len({r['sample_id'] for r in group})}
        for column in SCORE_COLUMNS:
            values = [int(r[column]) for r in group if r.get(column, "").strip()]
            out[f"mean_{column}"] = f"{sum(values) / len(values):.3f}" if values else ""
        guesses = [r.get("attribution_guess", "").strip() for r in group
                   if r.get("attribution_guess", "").strip()]
        out["attribution_guesses"] = len(guesses)
        out["attribution_accuracy"] = (
            f"{sum(guess == author for guess in guesses) / len(guesses):.3f}"
            if guesses and author and author != "base_gpt2" else ""
        )
        result.append(out)

    output_path = directory / "human_rating_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote human rating summary to {output_path}.")


if __name__ == "__main__":
    main()
