"""Reload and evaluate saved policies; never constructs an optimizer."""
import hashlib
import json
from pathlib import Path
import torch
from training.model_factory import load_model_and_tokenizer
from training.types import Runtime
from training.configs import EvaluationConfig
from training.data.gsm8k import load_questions, load_prompt_template
from training.evaluation.generation import evaluate


def evaluate_checkpoint(checkpoint, validation, prompt, output, *, device="cuda:0", max_new_tokens=512,
                        examples=None, seed=42):
    """
    训练后 从JSONL读取问题并推理生产答案进行评分
    """
    checkpoint, validation, output = Path(checkpoint), Path(validation), Path(output)
    if output.exists():
        raise FileExistsError(output)
    questions = load_questions(validation)
    if examples is not None:
        questions = questions[:examples]
    model, tokenizer = load_model_and_tokenizer(checkpoint, dtype=torch.bfloat16, device=device, local_files_only=True)
    model.requires_grad_(False)
    config = EvaluationConfig(device=device, seed=seed, max_new_tokens=max_new_tokens)
    metrics = evaluate(Runtime(model, tokenizer, None, model), questions, load_prompt_template(Path(prompt)), config)
    selection = checkpoint / "selection.json"
    report = {"checkpoint": str(checkpoint), "dtype": "bfloat16", "device": device,
        "validation": str(validation), "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "num_examples": len(questions), "seed": seed, "max_new_tokens": max_new_tokens,
        "selection": json.loads(selection.read_text()) if selection.is_file() else None, "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report
