"""Export a local Qwen3.5 Base checkpoint's unchanged text backbone for training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if not (source / "config.json").is_file():
        parser.error("--source must contain a local config.json")
    if destination == source or (destination.exists() and any(destination.iterdir())):
        parser.error("--destination must be a separate empty directory")
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
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
