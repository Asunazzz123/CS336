"""A constant-size, model-only best-checkpoint cache.

Use a fresh directory on /dev/shm for RAM-backed storage. This generic module
does not assume a filesystem type, choose an evaluation split, or own training
state. A replacement temporarily requires BOTH old and candidate checkpoints.
"""
from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
from threading import RLock
from typing import Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


def _read_selection(directory: Path) -> dict:
    path = directory / "selection.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Selection metadata must be a regular file")
    selection = json.loads(path.read_text())
    if not isinstance(selection, dict):
        raise ValueError("Invalid best-checkpoint selection metadata")
    step, score = selection.get("step"), selection.get("score")
    if (not isinstance(step, int) or isinstance(step, bool) or step < 0
            or not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score)
            or selection.get("mode") not in ("min", "max")
            or not isinstance(selection.get("metric_name"), str) or not selection["metric_name"]):
        raise ValueError("Invalid best-checkpoint selection metadata")
    json.dumps(selection, allow_nan=False)
    return selection


def _export_checkpoint(root: Path, source: Path, destination) -> Path:
    """Shared copying/publication path; callers validate cache ownership first."""
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = destination.parent.resolve() / destination.name
    if destination == root or root in destination.parents:
        raise ValueError("Export destination must be outside the cache")
    validate_hf_checkpoint(source)
    selection = _read_selection(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".partial-" + uuid4().hex)
    try:
        shutil.copytree(source, temporary, symlinks=False)
        validate_hf_checkpoint(temporary)
        if _read_selection(temporary) != selection:
            raise RuntimeError("Checkpoint selection changed during export")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    except BaseException:
        try:
            if temporary.is_dir() and not temporary.is_symlink():
                shutil.rmtree(temporary)
            elif temporary.is_symlink():
                temporary.unlink()
        except OSError:
            logger.warning("Could not remove failed checkpoint export: %s", temporary, exc_info=True)
        raise
    return destination


def export_cached_best(root, destination) -> Path:
    """Recover a RAM-cached best after its writer has STOPPED, without mutation.

    No manager instance is reopened and no stale generations are removed.
    The ownership marker, direct-child best symlink, selection and model files
    must all be valid. Do not invoke concurrently with a training writer or
    another exporter targeting the same destination.
    """
    root = Path(root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Cache root must be a real directory")
    root = root.resolve()
    marker = root / ".best-store-owner.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Cache ownership marker is missing or unsafe")
    owner = json.loads(marker.read_text())
    if not isinstance(owner, dict) or not isinstance(owner.get("owner"), str) or not re.fullmatch(
        r"[0-9a-f]{32}", owner["owner"]
    ):
        raise ValueError("Invalid cache ownership marker")
    pointer = root / "best"
    if not pointer.is_symlink():
        raise ValueError("Cache has no best-checkpoint symlink")
    target = os.readlink(pointer)
    if not re.fullmatch(r"\.checkpoint-[0-9a-f]{32}", target):
        raise ValueError("Best pointer must name a direct checkpoint child")
    source = root / target
    if source.is_symlink() or not source.is_dir() or source.resolve().parent != root:
        raise ValueError("Best pointer must refer to a real direct checkpoint child")
    return _export_checkpoint(root, source, destination)


def validate_hf_checkpoint(directory: Path) -> None:
    """Check config, safetensors headers and all indexed names without loading weights.

    Only model-only safetensors checkpoints are supported. Symlinked files and
    training-state payloads are rejected so the published tree is self-contained.
    This validates structure/readability, not a cryptographic content checksum or
    a model-specific forward/reload test.
    """
    from safetensors import safe_open

    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Checkpoint callback must create a real directory")
    for entry in directory.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"Checkpoint must be self-contained; symlink found: {entry.name}")
        if not entry.is_file() and not entry.is_dir():
            raise ValueError("Checkpoint contains a special filesystem entry")
        if entry.is_file() and (entry.suffix in (".pt", ".pth")
                                or entry.name.startswith("optimizer")):
            raise ValueError("Best checkpoint must be model-only; training-state payload found")
    config = json.loads((directory / "config.json").read_text())
    if not isinstance(config, dict) or not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise ValueError("Checkpoint config.json must contain a model_type")
    single = directory / "model.safetensors"
    index = directory / "model.safetensors.index.json"
    if single.exists() and index.exists():
        raise ValueError("Ambiguous checkpoint: both single model and sharded index exist")
    if index.is_file():
        payload = json.loads(index.read_text())
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Safetensors index requires a nonempty weight_map")
        expected = {}
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not name or not isinstance(filename, str):
                raise ValueError("Invalid safetensors weight mapping")
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
                raise ValueError("Shard paths must stay inside the checkpoint")
            expected.setdefault(relative, set()).add(name)
    elif single.is_file():
        expected = {Path("model.safetensors"): None}
    else:
        raise ValueError("Checkpoint has no safetensors model or index")
    actual_files = {path.relative_to(directory) for path in directory.rglob("*.safetensors")}
    if actual_files != set(expected):
        raise ValueError("Missing or unreferenced safetensors shards")
    for filename, expected_names in expected.items():
        with safe_open(str(directory / filename), framework="pt", device="cpu") as weights:
            names = set(weights.keys())
            if not names or (expected_names is not None and names != expected_names):
                raise ValueError(f"Safetensors names do not match index: {filename}")


class BestCheckpointStore:
    """Single-writer rolling model checkpoint with strict min/max selection.

    root must be new or empty. consider's save(path) receives a path that does
    NOT yet exist and must write a complete HF model-only checkpoint there.
    Its dtype, tokenizer and provenance are the callback's responsibility.
    A cache on tmpfs counts against host/cgroup RAM and is not restart durable.
    No implicit cleanup occurs on task failure or process exit.
    """
    def __init__(self, root, metric_name: str, mode: str):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError("metric_name must be nonempty")
        self.root = Path(root).expanduser().absolute()
        if self.root.is_symlink():
            raise ValueError("Cache root must not be a symlink")
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or any(self.root.iterdir()):
            raise FileExistsError("Best checkpoint cache must be a new or empty directory")
        self.metric_name, self.mode = metric_name, mode
        self._token = uuid4().hex
        self._marker = self.root / ".best-store-owner.json"
        with self._marker.open("x") as file:
            json.dump({"owner": self._token}, file)
        self._selection = None
        self._current = None
        self._owned = set()
        self._closed = False
        self._lock = RLock()

    @property
    def best_path(self) -> Path:
        """Stable symlink path; it exists only after the first successful capture."""
        return self.root / "best"

    @property
    def selection(self) -> dict | None:
        with self._lock:
            return deepcopy(self._selection)

    def _assert_owned(self):
        if self._closed or self.root.is_symlink():
            raise RuntimeError("Best checkpoint store is closed or its root was replaced")
        if json.loads(self._marker.read_text()).get("owner") != self._token:
            raise RuntimeError("Best checkpoint cache ownership changed")
        if self._current is not None:
            if not self.best_path.is_symlink() or os.readlink(self.best_path) != self._current.name:
                raise RuntimeError("Best checkpoint pointer was changed externally")
            if self._current.is_symlink() or not self._current.is_dir():
                raise RuntimeError("Selected checkpoint directory was changed externally")
        elif self.best_path.exists() or self.best_path.is_symlink():
            raise RuntimeError("Refusing an unowned best-checkpoint pointer")

    def _remove(self, path):
        if path not in self._owned or path.parent != self.root:
            raise RuntimeError("Refusing to remove an unowned checkpoint")
        if path.is_symlink():
            path.unlink()  # Never follow an unexpected link during cleanup.
        elif path.exists():
            shutil.rmtree(path)
        self._owned.discard(path)

    def _prune(self, *, strict):
        for path in list(self._owned - {self._current}):
            try:
                self._remove(path)
            except OSError:
                if strict:
                    raise
                logger.warning("Best checkpoint published, but stale cache cleanup failed: %s", path,
                               exc_info=True)

    def consider(self, step: int, score: float, save: Callable[[Path], None],
                 metadata: dict | None = None) -> bool:
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("score must be finite")
        # Copy and validate before invoking a potentially expensive save callback.
        selection = json.loads(json.dumps({"step": step, "metric_name": self.metric_name,
                                           "score": float(score), "mode": self.mode,
                                           "metadata": metadata}, allow_nan=False))
        with self._lock:
            self._assert_owned()
            if self._selection is not None:
                previous = self._selection["score"]
                if not (score < previous if self.mode == "min" else score > previous):
                    return False
            # Do not accumulate failed cleanup generations over many updates.
            self._prune(strict=True)
            candidate = self.root / (".checkpoint-" + uuid4().hex)
            self._owned.add(candidate)
            temporary_link = self.root / (".best-link-" + uuid4().hex)
            try:
                save(candidate)
                validate_hf_checkpoint(candidate)
                (candidate / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
                temporary_link.symlink_to(candidate.name, target_is_directory=True)
                os.replace(temporary_link, self.best_path)
            except BaseException:
                temporary_link.unlink(missing_ok=True)
                try:
                    self._remove(candidate)
                except OSError:
                    logger.warning("Could not remove failed best-checkpoint candidate: %s", candidate,
                                   exc_info=True)
                raise
            self._current, self._selection = candidate, selection
            self._prune(strict=False)
            return True

    def export(self, destination) -> Path:
        """Copy the selected checkpoint, validate, then atomically publish.

        Destination must be outside this cache and must not already exist.
        Concurrent writers to the same destination are not supported.
        Failure leaves the cache and any previously existing destination intact.
        """
        with self._lock:
            self._assert_owned()
            if self._current is None:
                raise RuntimeError("No best checkpoint has been captured")
            if _read_selection(self._current) != self._selection:
                raise RuntimeError("Best selection metadata changed externally")
            return _export_checkpoint(self.root, self._current, destination)

    def cleanup(self) -> None:
        """Explicitly remove only this instance's generations, pointer and marker."""
        with self._lock:
            if self._closed:
                return
            self._assert_owned()
            if self.best_path.is_symlink():
                self.best_path.unlink()
            self._current = None
            self._prune(strict=True)
            self._marker.unlink()
            self._closed = True
            # A user-added file in the root is preserved, not recursively deleted.
            if not any(self.root.iterdir()):
                self.root.rmdir()
