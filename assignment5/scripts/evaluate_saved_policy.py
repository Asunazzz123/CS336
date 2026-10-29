"""Saved-policy evaluation CLI."""
import argparse
import json
from pathlib import Path
import sys
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

PROJECT = PROJECT_DIR
from training.evaluation.reports import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=PROJECT / "alignment/prompts/r1_zero.prompt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.examples is not None and args.examples <= 0:
        parser.error("Token/example limits must be positive")
    report = evaluate_checkpoint(args.checkpoint, args.validation, args.prompt, args.output, device=args.device,
        max_new_tokens=args.max_new_tokens, examples=args.examples, seed=args.seed)
    print(json.dumps({"output": str(args.output), "accuracy": report["metrics"]["accuracy"]}))


if __name__ == "__main__":
    main()
