# Human evaluation of generated style

## Purpose

Compare the five author-fine-tuned GPT-2 checkpoints with base GPT-2 using identical prompts and decoding settings. The blind packet hides which model produced each continuation. This is a small, exploratory human study, not an objective or final benchmark.

## Generate the packet

After training all five models for the selected fold, run from the project root:

```bash
python src/generate_style_eval_samples.py --fold 1
```

The script uses five shared prompts, two sampled continuations per prompt/model by default (60 total for five tuned models plus base), and a reproducible seed. Decoding uses 150 maximum new tokens, temperature 0.85, top-k 50, top-p 0.95, and repetition penalty 1.1. It writes `generated_samples.csv`, `human_eval_packet.csv`, `human_eval_key.csv`, and `human_ratings_template.csv` to `results/generation_baseline/`.

## Rate samples

Reviewers should use the blinded packet and should not open the key until ratings are complete. Ideally, use at least two reviewers; record a consistent anonymous reviewer ID for each person. Rate each continuation from 1 (poor) to 5 (strong) on:

- **Fluency:** grammaticality and readability.
- **Coherence:** whether the continuation maintains a sensible sequence of ideas/events.
- **Topic fit:** whether it develops the supplied prompt.
- **Author guess:** choose one of the five listed authors or `unsure`; make a guess only when there is a basis for it.
- **Confidence:** 1 (low) to 5 (high) in the author guess.

Use these exact author keys in the `attribution_guess` field: `agatha_christie`, `arthur_conan_doyle`, `charles_dickens`, `jane_austen`, `william_shakespeare`, or `unsure`.

Use one row per reviewer per sample in a copy of the template. Rate all samples, including base GPT-2, before looking at model identities. The base model has no intended author, so its author-guess accuracy is not computed; it remains a fluency/coherence/topic-fit reference. Author familiarity can affect guesses, so keep this limitation with any reported result.

## Summarize

Save completed ratings as `human_ratings_completed.csv` in the output directory, then run:

```bash
python src/summarize_generation_ratings.py --ratings results/generation_baseline/human_ratings_completed.csv
```

This reports mean rubric scores and, for fine-tuned models only, the fraction of author guesses matching the fine-tuned author. It treats reviewer rows as observations and does not correct for reviewer agreement or the repeated prompts; interpret the summary descriptively. Report sample count, reviewer count, prompt set, model/fold, and deviations from the settings alongside any numbers.

## Data handling

Generated samples, the blind key, and human ratings are ignored by Git. Keep the key private until review is complete. Track the protocol and scripts; do not commit generated passages or the key.
