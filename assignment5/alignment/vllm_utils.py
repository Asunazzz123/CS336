"""Compatibility alias; implementation lives in generation.vllm."""
import sys
from generation import vllm as _implementation
sys.modules[__name__] = _implementation
