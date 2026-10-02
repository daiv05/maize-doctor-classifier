"""Regression tests for Modal batch recovery after container preemption."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.modal import source_stability_train as train


class InspectSlotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "seed_123"
        self.scenario = "maize-diseases"
        self.seed = 123
        self.reference = {}

    def metadata(self, status="training", attempt_count=None):
        self.root.mkdir(parents=True, exist_ok=True)
        splits, _ = train.scenario_paths(self.scenario, self.seed)
        payload = {
            "scenario": self.scenario,
            "seed": self.seed,
            "status": status,
            "split_lock_sha256": train.LOCK_SHA256[self.scenario],
            "command": train.command(self.scenario, self.seed, splits, self.root),
        }
        if attempt_count is not None:
            payload["attempt_count"] = attempt_count
        if status == "validation_complete":
            payload["run_id"] = "run_1"
            payload["checkpoint_sha256"] = "hash_1"
        train.write_json(self.root / "experiment_metadata.json", payload)

    def make_run(self, complete=False):
        run_dir = self.root / "runs" / train.MODEL / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / ("summary.json" if complete else "last.pth")).write_text("partial")
        return run_dir

    def test_fresh_slot(self):
        self.assertEqual(
            train.inspect_slot(self.root, self.scenario, self.seed, self.reference),
            (False, 1),
        )

    def test_completed_slot_is_verified_and_skipped(self):
        self.metadata("validation_complete")
        self.make_run(complete=True)
        with patch.object(
            train, "verify_run", return_value={"run_id": "run_1", "checkpoint_sha256": "hash_1"}
        ) as verify:
            self.assertEqual(
                train.inspect_slot(self.root, self.scenario, self.seed, self.reference),
                (True, 1),
            )
            verify.assert_called_once()

    def test_completed_run_recovers_interrupted_metadata_update(self):
        self.metadata()
        self.make_run(complete=True)
        with patch.object(
            train, "verify_run", return_value={"run_id": "run_1", "checkpoint_sha256": "hash_1"}
        ):
            self.assertEqual(
                train.inspect_slot(self.root, self.scenario, self.seed, self.reference),
                (True, 1),
            )
        payload = json.loads((self.root / "experiment_metadata.json").read_text())
        self.assertEqual(payload["status"], "validation_complete")

    def test_partial_first_attempt_is_archived_before_retry(self):
        self.metadata()
        self.make_run()
        self.assertEqual(
            train.inspect_slot(self.root, self.scenario, self.seed, self.reference),
            (False, 2),
        )
        self.assertTrue(
            (
                self.root
                / "interrupted_attempts"
                / "attempt_1"
                / train.MODEL
                / "run_1"
                / "last.pth"
            ).exists()
        )
        self.assertFalse((self.root / "runs" / train.MODEL).exists())
        # A second restart between archive and training must not create attempt 3.
        self.assertEqual(
            train.inspect_slot(self.root, self.scenario, self.seed, self.reference),
            (False, 2),
        )

    def test_second_partial_attempt_stops_without_overwrite(self):
        self.metadata(attempt_count=2)
        run_dir = self.make_run()
        with self.assertRaisesRegex(RuntimeError, "Dos intentos incompletos"):
            train.inspect_slot(self.root, self.scenario, self.seed, self.reference)
        self.assertTrue(run_dir.exists())

    def test_untracked_run_is_not_overwritten(self):
        self.make_run()
        with self.assertRaises(FileExistsError):
            train.inspect_slot(self.root, self.scenario, self.seed, self.reference)


if __name__ == "__main__":
    unittest.main()
