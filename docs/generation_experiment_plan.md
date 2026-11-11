# Generation experiment plan

## Goal

Measure whether author-conditioned adaptation improves generated prose over giving the same pretrained model a few examples of the target author's writing. Keep GPT-2 as a historical reference, not the only model or the presumed winner.

## Experimental arms

| Arm | Base model | Adaptation | Purpose |
|---|---|---|---|
| Historical reference | `gpt2` | Full causal-LM fine-tuning, one model per author | Reproduce the original project approach with book-held-out evaluation. |
| Modern prompt control | `Qwen/Qwen3.5-0.8B-Base` | Frozen weights; provide training-book style examples at inference | Measure what in-context examples alone can do. |
| Modern adapted model | The same Qwen3.5-0.8B-Base weights | One author-specific LoRA adapter per author | Test whether learned author adapters improve on the matched prompt control. |

The key comparison is the two Qwen arms because they use the same starting checkpoint. GPT-2 differs in both model and adaptation method, so treat it as historical context rather than evidence isolating either factor. Add GPT-2 + LoRA only if results or an interview question make that extra control useful.

Qwen3.5-0.8B-Base is selected as the current small open-weight candidate because the official model card describes it as suitable for in-context learning and fine-tuning, and calls out LoRA-style PEFT support with its chat template. It has a hybrid text/multimodal architecture, which is extra complexity for a text-only project; begin with a tiny loading, forward-pass, generation, and adapter-training smoke run before committing to the full experiment. If the Colab software stack or PEFT path is unstable, fall back to the simpler Qwen3-0.6B-Base checkpoint and document why. Lock the exact model revision and library versions once that smoke run succeeds.

## Data split and leakage controls

- Use the existing complete-book folds. For fold *k*, train author models and build few-shot examples from that author's other books only; evaluate on the held-out book.
- Use fold 1 for the initial pipeline pilot. For a headline estimate, run all five folds after adding more balanced books per author as already planned. The ten current external-validation books remain untouched and are not training or prompt examples.
- Use a fixed epoch count for each pilot. The held-out fold book is scored only after training and must not be used to select a checkpoint or epoch; tune schedules on training-book-only inner splits if tuning is later needed.
- Keep the generation prompt topics fixed across arms. The prompt-control examples, LoRA training examples, and GPT-2 training text must come only from that fold's training books.
- Record the model revision, library versions, split IDs, random seeds, token/block settings, LoRA settings, decoding parameters, and hardware/runtime for every run.
- Public-domain text may have appeared in foundation-model pretraining. Book-held-out splitting prevents leakage from this project's own fine-tuning data; it cannot establish that pretrained models never saw a test work. State this limitation and do not interpret a low loss as proof that the model learned author style rather than recalled text.

## Evaluation

1. **Held-out likelihood:** score each model on the same held-out source text with its own tokenizer. Per-token perplexity is meaningful within a tokenizer/model setup, but should not be ranked directly between GPT-2 and Qwen.
2. **Normalized text likelihood:** also report byte-normalized negative log-likelihood on identical UTF-8 text as a supplementary comparison. This reduces the direct dependence on token count, but is not a perfect tokenizer-neutral measure; interpret alongside the other evidence.
3. **Cross-author fit:** for every author-specific fine-tune/adapter, score each held-out author's book to form a within-family cross-author matrix. This checks whether an author adapter favors its target author's held-out book, but does not measure readability or whether a generated continuation follows a new topic.
4. **Generation attribution proxy:** score generated passages with the authorship classifier trained only on that fold's training books. Report target-author attribution by model and include a proper-name-masked sensitivity analysis if feasible. This is a proxy for stylistic resemblance, can be gamed by lexical or content cues, and is not a standalone quality score.
5. **Small blind human review:** sample a balanced subset from each arm (initial target: two outputs per author per arm, 30 outputs for three arms) and ask two reviewers to rate fluency, coherence, topic fit, and author guess. Keep identity hidden until ratings are complete. Treat results as descriptive because the number of prompts, works, and reviewers is small.

Use identical prompts and decoding settings wherever the interface permits. Because GPT-2 and Qwen have different tokenizers and different prompting conventions, describe results as a controlled project comparison, not as a universal model leaderboard. A prompt-only Qwen arm with the same Qwen base checkpoint is the most informative control for the question “does author adaptation help?”

## Execution order

1. Prepare a separate generation environment from `requirements-generation.txt` so model-support upgrades do not destabilize the existing authorship pipeline. It pins Transformers 5.22.2, PEFT 0.21.2, and Accelerate 1.15.0 while requiring PyTorch 2.5 or newer; use a runtime-appropriate PyTorch build.
2. Run `src/smoke_test_modern_generation.py` on a GPU runtime. It checks model loading, finite causal-LM loss, `all-linear` LoRA injection, finite gradients and one synthetic optimizer step, adapter save/reload, and text generation. This is a compatibility check, not evidence of learning or an evaluation result.
3. Run the GPT-2 fold-1 baseline as a pipeline pilot and time the complete run. Do not promise a fixed free-Colab runtime.
4. If the smoke test succeeds, lock the Qwen3.5 model revision and record package versions/device in the experiment metadata. If its hybrid architecture or PEFT target discovery is unstable, fall back to Qwen3-0.6B-Base and document the change.
5. Run `src/run_qwen_generation_experiment.py --mode all --fold 1 --epochs 1` as the initial Qwen pilot. It trains one fixed epoch, scores the base and five author adapters on the held-out books, generates two modern arms on shared topics, and creates a balanced blind packet. Do not interpret the one-epoch pilot as a tuned result.
6. Add fold-specific attribution scoring and collect the small blind human subset.
7. Expand the corpus, then run all book folds and reserve newly acquired books for a fresh final external check before publishing resume figures.

## Resume reporting rule

Report the data split and book counts with every score. Keep the current classifier findings explicitly provisional until corpus expansion and fresh external validation are complete. Generation claims should state the number of authors, books, prompts, samples, reviewers, and whether the result is held-out-book or external. Do not claim that the newer model is better unless the measured evaluation supports it.
