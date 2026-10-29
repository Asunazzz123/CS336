"""Structured metric emission, without trainer dependencies."""
import json
from utils.serialization import json_value


def log_metrics(output_dir, step, split, metrics):
    record = json_value({"step": step, "split": split, **metrics})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "examples"}, ensure_ascii=False, allow_nan=False), flush=True)
