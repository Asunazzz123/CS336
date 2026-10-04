from unittest.mock import patch

import pytest
import torch

from train.performance import PerformanceProbe


def test_warmup_excluded_and_weighted_throughput():
    probe = PerformanceProbe(torch.device("cpu"), warmup_steps=1)
    with patch("train.performance.time.perf_counter", return_value=100.0):
        probe.end_step(0.0, 0, 16, {"data_loading": 90.0})
    with patch("train.performance.time.perf_counter", return_value=2.0):
        probe.end_step(1.0, 1, 16, {"data_loading": 0.2})
    with patch("train.performance.time.perf_counter", return_value=5.0):
        probe.end_step(2.0, 2, 16, {"data_loading": 0.4})
    result = probe.summary()
    assert result["measured_steps"] == 2
    assert result["warmup_steps_excluded"] == 1
    assert result["measured_tokens"] == 32
    assert result["tokens_per_second"] == 8.0
    assert result["mean_step_seconds"] == 2.0
    assert result["std_step_seconds"] == 1.0
    assert result["mean_phase_seconds"]["data_loading"] == pytest.approx(0.3)
    assert result["peak_allocated_bytes"] is None


def test_no_measured_steps_reports_no_speed():
    result = PerformanceProbe(torch.device("cpu"), 5).summary()
    assert result["tokens_per_second"] is None
    assert result["mean_step_seconds"] is None
    assert result["mean_phase_seconds"] == {}


def test_phase_is_recorded_on_exception():
    probe = PerformanceProbe(torch.device("cpu"), 0)
    timings = {}
    with patch("train.performance.time.perf_counter", side_effect=[1.0, 1.25]):
        with pytest.raises(RuntimeError):
            with probe.phase(timings, "data_loading"):
                raise RuntimeError("test")
    assert timings == {"data_loading": 0.25}


def test_cuda_calls_use_selected_device():
    probe = PerformanceProbe(torch.device("cuda:1"), 1)
    with patch("torch.cuda.synchronize") as sync, patch("torch.cuda.reset_peak_memory_stats") as reset:
        with patch("torch.cuda.max_memory_allocated", return_value=1024), patch("torch.cuda.max_memory_reserved", return_value=2048):
            started = probe.begin_step(1)
            probe.end_step(started, 1, 16, {})
        reset.assert_called_once_with(torch.device("cuda:1"))
        assert all(call.args == (torch.device("cuda:1"),) for call in sync.call_args_list)
    result = probe.summary()
    assert result["peak_allocated_bytes"] == 1024
    assert result["peak_reserved_bytes"] == 2048
