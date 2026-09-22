import torch

from src.data.preparation import atomic_write_json
from src.data.transforms import CornTransformFactory
from src.training.loop import fit
from src.training.runs import (
    CheckpointIntegrityError,
    build_run_contract,
    load_validated_run,
    resolve_run_dir,
    write_latest_pointer,
    write_run_contract,
)


def test_e2e_checkpoint_contractual_corresponde_a_best_epoch(tmp_path, monkeypatch):
    run_dir = tmp_path / "tiny" / "e2e"
    run_dir.mkdir(parents=True)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for name in ("train", "val", "test"):
        (splits_dir / f"{name}.csv").write_text("sample_id,image_path,label\n")
    atomic_write_json(splits_dir / "manifest.lock.json", {"schema_version": 1})

    model = torch.nn.Linear(1, 1, bias=False)
    current_epoch = [0]

    def controlled_epoch(model, loader, criterion, device, optimizer=None, **kwargs):
        del loader, criterion, device, kwargs
        if optimizer is not None:
            current_epoch[0] += 1
            with torch.no_grad():
                model.weight.fill_(current_epoch[0])
        score = [0.9, 0.7, 0.1][current_epoch[0] - 1]
        return {"loss": 1.0, "accuracy": score, "macro_f1": score}, [], [], []

    monkeypatch.setattr("src.training.loop.run_epoch", controlled_epoch)
    history = fit(
        model=model,
        train_loader=[],
        val_loader=[],
        criterion=None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        epochs=3,
        model_name="tiny",
        run_dir=run_dir,
    )
    assert model.weight.item() == 1.0
    assert torch.load(run_dir / "last.pth", weights_only=True)["weight"].item() == 3.0

    factory = CornTransformFactory(target_size=(8, 8))
    contract = build_run_contract(
        run_dir=run_dir,
        model_name="tiny",
        seed=42,
        hyperparameters={
            "learning_rate": 0.1,
            "batch_size": 1,
            "weight_decay": 0.0,
            "optimizer": "SGD",
            "scheduler": "none",
            "epochs": 3,
            "patience": None,
            "dropout": None,
        },
        preprocessing=factory.to_contract(),
        training_preprocessing=factory.training_contract(),
        class_to_idx={"a": 0},
        splits_dir=splits_dir,
        best_epoch=1,
        metrics={"best_validation": {"epoch": 1, "macro_f1": 0.9}},
    )
    write_run_contract(run_dir, contract)
    write_latest_pointer(tmp_path, "tiny", "e2e")
    assert resolve_run_dir(tmp_path, "tiny") == run_dir
    monkeypatch.setattr(
        "src.training.runs.build_model",
        lambda *args, **kwargs: torch.nn.Linear(1, 1, bias=False),
    )

    loaded = load_validated_run(run_dir / "best.pth", expected_model="tiny")
    assert loaded.model.weight.item() == 1.0
    assert loaded.model(torch.tensor([[2.0]])).item() == 2.0
    state = json_load(run_dir / "training_state.json")
    assert state == {
        "best_epoch": 1,
        "best_metric": 0.9,
        "last_epoch": 3,
        "schema_version": 1,
        "status": "complete",
    }
    assert history[-1]["epoch"] == 3

    with (run_dir / "best.pth").open("ab") as handle:
        handle.write(b"tamper")
    try:
        load_validated_run(run_dir / "best.pth", expected_model="tiny")
    except CheckpointIntegrityError:
        pass
    else:  # pragma: no cover - mensaje más claro que un assert genérico
        raise AssertionError("El checkpoint alterado debía ser rechazado.")


def json_load(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
