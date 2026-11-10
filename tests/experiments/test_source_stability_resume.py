"""Recuperación por lote de Modal tras preempción de contenedor."""

import json
from types import SimpleNamespace

import pytest

from scripts.modal import source_stability_train as train

SCENARIO = "maize-diseases"
SEED = 123
VERIFIED = {"run_id": "run_1", "checkpoint_sha256": "hash_1"}


@pytest.fixture
def slot(tmp_path):
    root = tmp_path / "seed_123"

    def write_metadata(status="training", attempt_count=None):
        root.mkdir(parents=True, exist_ok=True)
        splits, _ = train.scenario_paths(SCENARIO, SEED)
        payload = {
            "scenario": SCENARIO,
            "seed": SEED,
            "status": status,
            "split_lock_sha256": train.LOCK_SHA256[SCENARIO],
            "command": train.command(SCENARIO, SEED, splits, root),
        }
        if attempt_count is not None:
            payload["attempt_count"] = attempt_count
        if status == "validation_complete":
            payload["run_id"] = "run_1"
            payload["checkpoint_sha256"] = "hash_1"
        train.write_json(root / "experiment_metadata.json", payload)

    def make_run(complete=False):
        run_dir = root / "runs" / train.MODEL / "run_1"
        run_dir.mkdir(parents=True)
        name = "summary.json" if complete else "last.pth"
        (run_dir / name).write_text("partial", encoding="utf-8")
        return run_dir

    def inspect():
        return train.inspect_slot(root, SCENARIO, SEED, {})

    return SimpleNamespace(
        root=root, write_metadata=write_metadata, make_run=make_run, inspect=inspect
    )


def test_fresh_slot(slot):
    assert slot.inspect() == (False, 1)


def test_completed_slot_is_verified_and_skipped(slot, monkeypatch):
    slot.write_metadata("validation_complete")
    slot.make_run(complete=True)
    calls = []
    monkeypatch.setattr(train, "verify_run", lambda *args: calls.append(args) or VERIFIED)
    assert slot.inspect() == (True, 1)
    assert len(calls) == 1


def test_completed_run_recovers_interrupted_metadata_update(slot, monkeypatch):
    slot.write_metadata()
    slot.make_run(complete=True)
    monkeypatch.setattr(train, "verify_run", lambda *args: VERIFIED)
    assert slot.inspect() == (True, 1)
    payload = json.loads((slot.root / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert payload["status"] == "validation_complete"


def test_partial_first_attempt_is_archived_before_retry(slot):
    slot.write_metadata()
    slot.make_run()
    assert slot.inspect() == (False, 2)
    archived = slot.root / "interrupted_attempts" / "attempt_1" / train.MODEL / "run_1" / "last.pth"
    assert archived.exists()
    assert not (slot.root / "runs" / train.MODEL).exists()
    assert slot.inspect() == (False, 2)


def test_second_partial_attempt_stops_without_overwrite(slot):
    slot.write_metadata(attempt_count=2)
    run_dir = slot.make_run()
    with pytest.raises(RuntimeError, match="Dos intentos incompletos"):
        slot.inspect()
    assert run_dir.exists()


def test_untracked_run_is_not_overwritten(slot):
    slot.make_run()
    with pytest.raises(FileExistsError):
        slot.inspect()
