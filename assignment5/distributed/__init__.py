"""NVIDIA FSDP2 systems layer, intentionally independent of existing trainers.

The distributed package describes the systems responsibility; NVIDIA is the backend.
Importing this package does not initialize CUDA or distributed communication.
"""
import torch

if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 10):
    raise RuntimeError("distributed requires PyTorch >=2.10")

from .config import FSDPConfig, GenerationConfig
from .context import NvidiaContext
from .engine import NvidiaFSDPEngine, LossOutput, StepResult, RolloutResult, local_hf_factory
from .state import Phase
from .weights import WeightSink, HFCheckpointSink, export_model_only_hf
from .best_checkpoint import DistributedBestCheckpoint

__all__ = ["FSDPConfig", "GenerationConfig", "NvidiaContext", "NvidiaFSDPEngine",
           "LossOutput", "StepResult", "RolloutResult", "local_hf_factory", "Phase",
           "WeightSink", "HFCheckpointSink", "export_model_only_hf", "DistributedBestCheckpoint"]
