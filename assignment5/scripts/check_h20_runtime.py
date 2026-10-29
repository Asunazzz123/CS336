"""H20 diagnostic CLI; does not run when imported."""
import argparse
import os
from pathlib import Path
import sys
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Existing local text-only model directory; never downloaded here")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8017)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-update", action="store_true",
                        help="Only load, synchronize, score and generate; no backward/optimizer step")
    parser.add_argument("--verify-weight-transfer", action="store_true",
                        help="Read worker parameter samples through an optional RPC and verify restored sentinels")
    args = parser.parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
        parser.error("--model-path must be an existing local model directory containing config.json")
    if args.report == args.model_path or args.model_path in args.report.parents:
        parser.error("--report must be outside the model directory")
    if args.report.suffix != ".json" or args.report.exists():
        parser.error("--report must be a new .json file")
    if not 1 <= args.max_new_tokens <= 32:
        parser.error("--max-new-tokens must be between 1 and 32 for this short acceptance check")
    if args.verify_weight_transfer and args.generation_backend != "vllm":
        parser.error("--verify-weight-transfer requires the vllm backend")
    return args



def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from diagnostics.h20_runtime import run_checks
    run_checks(args)


if __name__ == "__main__":
    main()
