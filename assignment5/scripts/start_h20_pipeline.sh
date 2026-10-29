#!/usr/bin/env bash
set -e
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec bash "$PROJECT_DIR/orchestration/sft_rl_pipeline.sh" "$@"
