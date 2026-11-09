"""Smoke-test Qwen3.5 text generation and LoRA save/reload integration.

This downloads the selected checkpoint and performs one tiny synthetic training
step. It does not read project books or produce evaluation scores.
"""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"
ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "generation_baseline" / "qwen35_smoke_summary.json"
SMOKE_TEXT = (
    "The rain had stopped before the last carriage reached the village. "
    "A lamp was burning in the window, though the house appeared empty."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main", help="Use a commit hash for reproducible runs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def choose_device(name: str, torch):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if name == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("Apple MPS was requested but is unavailable.")
    return torch.device(name)


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    import peft
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    major, minor = (int(part) for part in torch_version.split(".")[:2])
    if (major, minor) < (2, 5):
        raise RuntimeError(
            f"This isolated environment needs PyTorch >=2.5; found {torch.__version__}. "
            "Do not upgrade the main project environment. Use a separate venv or Colab runtime."
        )

    device = choose_device(args.device, torch)
    if device.type == "cpu":
        print("Warning: CPU mode can be slow; use a GPU runtime for the full smoke check.")
    print(
        f"model={args.model} revision={args.revision} device={device} "
        f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Selected tokenizer has no EOS token; cannot configure causal-LM padding.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FP32 is intentional for this tiny one-step compatibility check. It avoids
    # conflating adapter support with low-precision overflow on a T4-class GPU.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.float32,
    ).to(device)
    model.config.use_cache = False
    encoded = tokenizer(
        SMOKE_TEXT,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        initial = model(**encoded, labels=encoded["input_ids"])
    if initial.loss is None or not torch.isfinite(initial.loss):
        raise RuntimeError("Base-model forward pass returned a non-finite loss.")
    print(f"Base-model forward pass succeeded (loss={initial.loss.item():.4f}).")
    del initial

    # all-linear adapts attention and feed-forward projections while leaving
    # the large token embedding/output matrix frozen.
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if not trainable or trainable_count >= total_count:
        raise RuntimeError("LoRA did not create a smaller, non-empty trainable parameter set.")
    model.print_trainable_parameters()

    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(**encoded, labels=encoded["input_ids"])
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("LoRA training forward pass returned a non-finite loss.")
    output.loss.backward()
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("LoRA backward pass produced missing or non-finite gradients.")
    optimizer.step()
    print(f"One synthetic LoRA optimizer step succeeded (loss={output.loss.item():.4f}).")

    model.config.use_cache = True
    model.eval()
    with tempfile.TemporaryDirectory(prefix="wsm_qwen_lora_smoke_") as adapter_dir:
        model.save_pretrained(adapter_dir)
        del model, optimizer, trainable, gradients, output
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.float32,
        ).to(device)
        reloaded = PeftModel.from_pretrained(base, adapter_dir).eval()
        with torch.inference_mode():
            generated = reloaded.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if generated.shape[1] <= encoded["input_ids"].shape[1]:
            raise RuntimeError("Reloaded adapter did not generate any continuation tokens.")
        print("Adapter save/reload and text generation succeeded.")
        del reloaded, base, generated

    summary = {
        "status": "passed",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(tokenizer, "_commit_hash", None),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": str(device),
        "dtype": "float32",
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "synthetic_one_step_only": True,
        "project_corpus_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved smoke-test environment summary to {args.output}.")


if __name__ == "__main__":
    main()
