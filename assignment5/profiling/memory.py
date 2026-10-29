"""Device/process memory observations; sampling is not exact kernel timing."""
import csv
import resource
import subprocess
import sys
import threading
from pathlib import Path
import torch


def nvidia_smi_snapshot():
    fields = "index,uuid,name,memory.used,memory.total"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10, check=True,
        )
        return [dict(zip(fields.split(","), [field.strip() for field in row]))
                for row in csv.reader(result.stdout.splitlines())]
    except (OSError, subprocess.SubprocessError) as error:
        return {"unavailable": str(error)}


class GPUMemoryMonitor:
    """Sample device-wide usage, which includes the separate vLLM process.

    Torch allocator statistics in this process cannot measure vLLM's GPU1
    allocations. These sampled device peaks are observations, not exact peaks.
    """

    def __init__(self, interval=1.0):
        self.interval = interval
        self.peaks = {}
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            rows = nvidia_smi_snapshot()
            if isinstance(rows, list):
                for row in rows:
                    try:
                        used = float(row["memory.used"])
                    except ValueError:
                        continue
                    self.peaks[row["uuid"]] = max(self.peaks.get(row["uuid"], 0), used)
                self.samples += 1
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=12)


def memory_snapshot():
    snapshot = {"nvidia_smi_devices": nvidia_smi_snapshot(), "torch_allocator": []}
    for index in range(torch.cuda.device_count()):
        snapshot["torch_allocator"].append({
            "device": index,
            "allocated_bytes": torch.cuda.memory_allocated(index),
            "reserved_bytes": torch.cuda.memory_reserved(index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        })
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM"):
                snapshot[name + "_bytes"] = int(value.split()[0]) * 1024
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        snapshot["VmHWM_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    snapshot["cgroup"] = {}
    for filename in ("memory.current", "memory.peak", "memory.max"):
        path = Path("/sys/fs/cgroup") / filename
        if path.exists():
            value = path.read_text().strip()
            snapshot["cgroup"][filename] = int(value) if value.isdigit() else value
    return snapshot
