"""Best-checkpoint retention, failure atomicity and real HF model roundtrips."""
from pathlib import Path
import json

import pytest
import torch
from safetensors.torch import save_file, load_file

from checkpointing.best import BestCheckpointStore, export_cached_best, validate_hf_checkpoint


def checkpoint_writer(value):
    def write(path):
        assert not path.exists()  # Supports save_pretrained and external transactional writers.
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model_type": "tiny-test"}))
        save_file({"weight": torch.full((2, 3), value, dtype=torch.float32)}, str(path / "model.safetensors"))
    return write


def weight(store):
    return load_file(store.best_path / "model.safetensors")["weight"]


def generations(store):
    return list(store.root.glob(".checkpoint-*"))


@pytest.mark.parametrize("mode,first,better,worse", [("max", 0.4, 0.7, 0.3), ("min", 0.4, 0.2, 0.6)])
def test_strict_selection_and_constant_retention(tmp_path, mode, first, better, worse):
    store = BestCheckpointStore(tmp_path / "cache", "validation", mode)
    assert store.selection is None and not store.best_path.exists()
    with pytest.raises(RuntimeError, match="No best"):
        store.export(tmp_path / "out")
    assert store.consider(0, first, checkpoint_writer(1), metadata={"precision": "float32"})
    old_target = store.best_path.resolve()
    never_save = lambda p: pytest.fail("Non-improvement must not save another checkpoint")
    assert not store.consider(1, first, never_save)
    assert not store.consider(2, worse, never_save)
    assert store.consider(3, better, checkpoint_writer(2))
    assert not old_target.exists()
    assert len(generations(store)) == 1
    assert torch.equal(weight(store), torch.full((2, 3), 2.0))
    assert store.selection["step"] == 3
    for step in range(4, 20):
        score = better + (step if mode == "max" else -step)
        assert store.consider(step, score, checkpoint_writer(step))
        assert len(generations(store)) == 1
    selection = store.selection
    selection["step"] = -1
    assert store.selection["step"] == 19  # Returned metadata is not a mutable internal alias.


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), True])
def test_invalid_scores_do_not_capture(tmp_path, score):
    store = BestCheckpointStore(tmp_path / "cache", "accuracy", "max")
    with pytest.raises(ValueError, match="finite"):
        store.consider(1, score, lambda p: pytest.fail("Invalid score captured"))
    assert store.selection is None and not generations(store)


def test_failed_write_keeps_old_best_score_pointer_and_files(tmp_path):
    store = BestCheckpointStore(tmp_path / "cache", "accuracy", "max")
    store.consider(1, 0.5, checkpoint_writer(3))
    old = store.best_path.resolve()

    def failed(path):
        path.mkdir()
        (path / "partial").write_text("incomplete")
        raise OSError("simulated RAM full")

    with pytest.raises(OSError, match="RAM full"):
        store.consider(2, 0.7, failed)
    assert store.best_path.resolve() == old
    assert store.selection["step"] == 1
    assert len(generations(store)) == 1
    assert torch.equal(weight(store), torch.full((2, 3), 3.0))


@pytest.mark.parametrize("kind", ["truncated", "missing-shard", "bad-index-name", "training-state", "symlink"])
def test_invalid_candidates_preserve_old_best(tmp_path, kind):
    store = BestCheckpointStore(tmp_path / "cache", "accuracy", "max")
    store.consider(1, 0.5, checkpoint_writer(3))
    old = store.best_path.resolve()
    external = tmp_path / "external"
    external.write_text("do not delete")

    def invalid(path):
        checkpoint_writer(7)(path)
        if kind == "truncated":
            (path / "model.safetensors").write_bytes(b"bad")
        elif kind in ("missing-shard", "bad-index-name"):
            (path / "model.safetensors").unlink()
            name = "part.safetensors" if kind == "missing-shard" else "../external.safetensors"
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": name}}))
        elif kind == "training-state":
            torch.save({"optimizer": {}}, path / "training_state.pt")
        elif kind == "symlink":
            (path / "external").symlink_to(external)

    with pytest.raises(Exception):
        store.consider(2, 0.7, invalid)
    assert store.best_path.resolve() == old and store.selection["score"] == 0.5
    assert len(generations(store)) == 1
    assert external.read_text() == "do not delete"


def test_sharded_index_is_validated_against_actual_tensor_names(tmp_path):
    path = tmp_path / "sharded"
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"tiny-test"}')
    save_file({"a": torch.ones(2)}, str(path / "part-1.safetensors"))
    save_file({"b": torch.ones(2)}, str(path / "part-2.safetensors"))
    index = {"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    validate_hf_checkpoint(path)
    index["weight_map"]["wrong-name"] = index["weight_map"].pop("a")
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="names"):
        validate_hf_checkpoint(path)


def test_export_failure_keeps_cache_and_never_replaces_existing_data(tmp_path, monkeypatch):
    store = BestCheckpointStore(tmp_path / "cache", "accuracy", "max")
    store.consider(1, 0.5, checkpoint_writer(3))
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "important").write_text("preserve")
    with pytest.raises(FileExistsError):
        store.export(existing)
    assert (existing / "important").read_text() == "preserve"
    import checkpointing.best as module
    original_copy = module.shutil.copytree

    def failed_copy(source, destination, **kwargs):
        Path(destination).mkdir()
        (Path(destination) / "partial").write_text("incomplete")
        raise OSError("SSD full")

    monkeypatch.setattr(module.shutil, "copytree", failed_copy)
    with pytest.raises(OSError, match="SSD full"):
        store.export(tmp_path / "failed-export")
    assert not (tmp_path / "failed-export").exists()
    assert not list(tmp_path.glob(".failed-export.partial-*"))
    assert store.selection["score"] == 0.5 and store.best_path.is_dir()
    monkeypatch.setattr(module.shutil, "copytree", original_copy)
    exported = store.export(tmp_path / "exported")
    assert exported.is_dir() and not exported.is_symlink()
    assert json.loads((exported / "selection.json").read_text())["step"] == 1
    assert torch.equal(load_file(exported / "model.safetensors")["weight"], weight(store))
    assert store.best_path.is_dir()  # Successful export does not implicitly discard volatile cache.


def test_new_or_empty_root_and_cleanup_only_owned_paths(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "user-file").write_text("preserve")
    with pytest.raises(FileExistsError):
        BestCheckpointStore(root, "accuracy", "max")
    (root / "user-file").unlink()
    store = BestCheckpointStore(root, "accuracy", "max")
    store.consider(1, 0.5, checkpoint_writer(3))
    (root / "user-file").write_text("preserve")
    store.cleanup()
    store.cleanup()
    assert (root / "user-file").read_text() == "preserve"
    assert not generations(store) and not store.best_path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        store.consider(2, 0.7, checkpoint_writer(7))


def test_export_rejects_paths_inside_cache_even_through_parent_alias(tmp_path):
    store = BestCheckpointStore(tmp_path / "cache", "accuracy", "max")
    store.consider(1, 0.5, checkpoint_writer(3))
    alias = tmp_path / "alias"
    alias.symlink_to(store.best_path, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        store.export(alias / "bad-export")


def test_cleanup_refuses_a_user_added_best_pointer(tmp_path):
    store = BestCheckpointStore(tmp_path / "cache", "accuracy", "max")
    external = tmp_path / "external"
    external.mkdir()
    store.best_path.symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unowned"):
        store.cleanup()
    assert store.best_path.is_symlink() and external.is_dir()


def test_frozen_tied_hf_model_can_be_reloaded_after_live_model_changes(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config(vocab_size=9, n_layer=1, n_head=2, n_embd=8, n_positions=16))
    store = BestCheckpointStore(tmp_path / "cache", "loss", "min")
    expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    store.consider(3, 0.4, lambda path: model.save_pretrained(path))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    output = store.export(tmp_path / "best-model")
    restored = GPT2LMHeadModel.from_pretrained(output, local_files_only=True)
    assert restored.transformer.wte.weight.data_ptr() == restored.lm_head.weight.data_ptr()
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert store.selection["step"] == 3


def cache_contents(root):
    return {str(path.relative_to(root)): ("link", str(path.readlink())) if path.is_symlink()
            else ("file", path.read_bytes()) if path.is_file() else ("directory", None)
            for path in root.rglob("*")}


def test_offline_export_recovers_last_valid_model_after_writer_failure_without_cache_changes(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config(vocab_size=9, n_layer=1, n_head=2, n_embd=8, n_positions=16))
    expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    root = tmp_path / "cache"
    store = BestCheckpointStore(root, "loss", "min")
    store.consider(4, 0.4, lambda path: model.save_pretrained(path))

    def fail(path):
        path.mkdir()
        (path / "partial").write_text("interrupted")
        raise OSError("writer failed")

    with pytest.raises(OSError, match="writer failed"):
        store.consider(5, 0.3, fail)
    # The process can lose all Python manager state; only its directory remains.
    del store
    before = cache_contents(root)
    with pytest.raises(FileExistsError):
        BestCheckpointStore(root, "loss", "min")
    output = export_cached_best(root, tmp_path / "recovered")
    assert cache_contents(root) == before
    assert json.loads((output / "selection.json").read_text())["step"] == 4
    restored = GPT2LMHeadModel.from_pretrained(output, local_files_only=True)
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["outside", "indirect-child", "bad-selection", "missing-owner"])
def test_offline_export_rejects_unsafe_cache_metadata_and_links(tmp_path, kind):
    store = BestCheckpointStore(tmp_path / "cache", "accuracy", "max")
    store.consider(1, 0.5, checkpoint_writer(3))
    external = tmp_path / "external"
    checkpoint_writer(9)(external)
    if kind == "outside":
        store.best_path.unlink()
        store.best_path.symlink_to(external, target_is_directory=True)
    elif kind == "indirect-child":
        real = store.best_path.resolve()
        store.best_path.unlink()
        indirect = store.root / (".checkpoint-" + "a" * 32)
        indirect.symlink_to(real, target_is_directory=True)
        store.best_path.symlink_to(indirect.name, target_is_directory=True)
    elif kind == "bad-selection":
        payload = json.loads((store.best_path / "selection.json").read_text())
        payload["score"] = float("nan")
        (store.best_path / "selection.json").write_text(json.dumps(payload))
    else:
        (store.root / ".best-store-owner.json").unlink()
    before = cache_contents(store.root)
    with pytest.raises(ValueError):
        export_cached_best(store.root, tmp_path / "unsafe-export")
    assert cache_contents(store.root) == before
    assert not (tmp_path / "unsafe-export").exists()
    assert load_file(external / "model.safetensors")["weight"].eq(9).all()


def test_offline_export_write_failure_leaves_cache_and_existing_destination_intact(tmp_path, monkeypatch):
    store = BestCheckpointStore(tmp_path / "cache", "accuracy", "max")
    store.consider(1, 0.5, checkpoint_writer(3))
    before = cache_contents(store.root)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "user-file").write_text("preserve")
    with pytest.raises(FileExistsError):
        export_cached_best(store.root, existing)
    import checkpointing.best as module

    def failed_copy(source, destination, **kwargs):
        Path(destination).mkdir()
        (Path(destination) / "partial").write_text("incomplete")
        raise OSError("SSD full")

    monkeypatch.setattr(module.shutil, "copytree", failed_copy)
    with pytest.raises(OSError, match="SSD full"):
        export_cached_best(store.root, tmp_path / "failed-export")
    assert cache_contents(store.root) == before
    assert not (tmp_path / "failed-export").exists()
    assert not list(tmp_path.glob(".failed-export.partial-*"))
    assert (existing / "user-file").read_text() == "preserve"
