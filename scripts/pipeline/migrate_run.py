"""Migra explícitamente un run legacy al contrato versionado v1.

La herramienta nunca infiere preprocessing ni hiperparámetros ausentes. Cuando el
summary histórico no los contiene, deben entregarse como JSON revisado mediante las
banderas correspondientes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.runs import migrate_legacy_run


def _read_json(path: str | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} debe contener un objeto JSON.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directorio del run legacy.")
    parser.add_argument(
        "--preprocessing-contract",
        default=None,
        help="JSON revisado del preprocessing; obligatorio si el legacy no lo registró.",
    )
    parser.add_argument(
        "--hyperparameters",
        default=None,
        help="JSON con valores efectivos; obligatorio si el legacy no los registró.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Directorio verificable de splits; sobrescribe la ruta histórica.",
    )
    args = parser.parse_args()

    contract = migrate_legacy_run(
        args.run_dir,
        preprocessing=_read_json(args.preprocessing_contract),
        hyperparameters=_read_json(args.hyperparameters),
        seed=args.seed,
        splits_dir=args.splits_dir,
    )
    print(f"Run migrado: {contract['run_id']} | checkpoint_sha256={contract['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
