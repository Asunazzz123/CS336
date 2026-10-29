"""Unchanged BF16 text-backbone conversion with local provenance."""
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_text_checkpoint(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not (source / "config.json").is_file():
        raise ValueError("--source must contain a local config.json")
    if destination == source or (destination.exists() and any(destination.iterdir())):
        raise ValueError("--destination must be a separate empty directory")
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, local_files_only=True,
    )
    if model.config.model_type != "qwen3_5_text":
        raise ValueError(f"Expected Qwen3.5 text backbone, got {model.config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, max_shard_size="5GB")
    tokenizer.save_pretrained(destination)
    files = []
    for path in sorted(destination.glob("*.safetensors")):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
    report = {
        "source": str(source), "destination": str(destination),
        "operation": "Unchanged BF16 text-backbone export; no training or quantization",
        "model_class": type(model).__name__, "model_type": model.config.model_type,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": str(next(model.parameters()).dtype), "weight_files": files,
    }
    verification = source / "download_verification.json"
    if verification.is_file():
        report["source_download"] = json.loads(verification.read_text())
    (destination / "text_export_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
