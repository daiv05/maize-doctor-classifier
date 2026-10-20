"""Pruebas stdlib del control previo; se pueden ejecutar sin PyTorch."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.checks.preflight_data_artifacts import Inputs, run_preflight, sha256_file


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class PreflightDataArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        dataset = root / "dataset"
        output = root / "outputs"
        splits = output / "splits" / "seed_42"
        splits.mkdir(parents=True)
        config = root / "dataset.yaml"
        config.write_text("dataset: synthetic\n", encoding="utf-8")
        python_bin = root / "venv" / "bin" / "python"
        python_bin.parent.mkdir(parents=True)
        python_bin.touch()

        samples = []
        for name, environment in (("a", "lab"), ("b", "real"), ("c", "lab"), ("d", "real")):
            relative = f"clean/healthy/{environment}/{name}.jpg"
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            samples.append({"image_path": relative, "sha256": sha256_file(path)})

        exclusions = root / "dataset_exclusions.csv"
        write_csv(exclusions, ["image_path", "sha256", "reason"], [
            {**samples[3], "reason": "excluded"}
        ])
        master_rows = [
            {**sample, "sample_id": hashlib.sha256(sample["image_path"].encode()).hexdigest()}
            for sample in samples[:3]
        ]
        write_csv(
            splits / "master_manifest.csv", ["sample_id", "image_path", "sha256"], master_rows
        )
        for name, sample in zip(("train", "val", "test"), master_rows):
            write_csv(splits / f"{name}.csv", ["sample_id", "image_path"], [sample])
        (splits / "preparation_audit.json").write_text("{}", encoding="utf-8")
        (splits / "split_audit_report.csv").write_text("metric,value\n", encoding="utf-8")

        lock = {
            "schema_version": 1,
            "config_sha256": sha256_file(config),
            "exclusions_sha256": sha256_file(exclusions),
            "split_parameters": {"classes": ["healthy"]},
            "master_manifest_sha256": sha256_file(splits / "master_manifest.csv"),
            **{f"{name}_sha256": sha256_file(splits / f"{name}.csv")
               for name in ("train", "val", "test")},
        }
        lock_path = splits / "manifest.lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        expected_lock = root / "expected_lock.json"
        expected_lock.write_bytes(lock_path.read_bytes())
        expected_summary = root / "expected_summary.json"
        expected_summary.write_text(json.dumps({"seed_42": {
            "manifest_lock_sha256": sha256_file(expected_lock),
            "train": 1, "val": 1, "test": 1,
        }}), encoding="utf-8")

        self.inputs = Inputs(
            dataset_root=dataset,
            output_root=output,
            splits_dir=splits,
            expected_lock=expected_lock,
            expected_summary=expected_summary,
            config_path=config,
            exclusions_path=exclusions,
            python_bin=python_bin,
        )

    def test_acepta_corpus_y_splits_congelados(self) -> None:
        result = run_preflight(self.inputs)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(all(check["status"] == "ok" for check in result["checks"]))

    def test_detecta_imagen_modificada_aunque_el_split_coincida(self) -> None:
        (self.inputs.dataset_root / "clean/healthy/lab/a.jpg").write_bytes(b"changed")
        result = run_preflight(self.inputs)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("contenido_clean", {
            check["name"] for check in result["checks"] if check["status"] == "fail"
        })

    def test_detecta_csv_diferente_del_lock(self) -> None:
        with (self.inputs.splits_dir / "train.csv").open("a", encoding="utf-8") as handle:
            handle.write("unexpected,path\n")
        result = run_preflight(self.inputs)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("identidad_split", {
            check["name"] for check in result["checks"] if check["status"] == "fail"
        })

    def test_reporta_drift_de_config_sin_reescribir_lock(self) -> None:
        self.inputs.config_path.write_text("dataset: changed\n", encoding="utf-8")
        result = run_preflight(self.inputs)
        self.assertEqual(result["status"], "ready_with_warnings")
        self.assertIn("config", {
            check["name"] for check in result["checks"] if check["status"] == "warning"
        })

    def test_quick_no_declara_verificacion_completa(self) -> None:
        result = run_preflight(self.inputs, verify_image_hashes=False)
        self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
