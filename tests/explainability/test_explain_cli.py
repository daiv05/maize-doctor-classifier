import argparse
import builtins
import logging

import pandas as pd
import pytest
import torch
import torch.nn as nn

import scripts.pipeline.explain as explain_cli
from scripts.pipeline.explain import RunContext, build_parser

_SUBCOMMANDS = {"visual", "fidelity", "errors", "compare", "global"}


def test_every_subcommand_is_registered():
    parser = build_parser()

    actions = [action for action in parser._subparsers._group_actions if hasattr(action, "choices")]

    assert _SUBCOMMANDS <= set(actions[0].choices)


@pytest.mark.parametrize("subcommand", sorted(_SUBCOMMANDS))
def test_common_flags_are_available_everywhere(subcommand):
    args = build_parser().parse_args(
        [subcommand, "--models", "shufflenet_v2_x1_0", "--output-dir", "outputs/main"]
    )

    assert args.command == subcommand
    assert args.models == ["shufflenet_v2_x1_0"]
    assert args.output_dir == "outputs/main"


def test_visual_accepts_a_single_image():
    args = build_parser().parse_args(["visual", "--image", "foto.jpg", "--output", "out.png"])

    assert args.image == "foto.jpg"
    assert args.output == "out.png"


def test_fidelity_accepts_sampling_overrides():
    args = build_parser().parse_args(["fidelity", "--sample-size", "50", "--num-samples", "500"])

    assert args.sample_size == 50
    assert args.num_samples == 500


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_visual_fails_fast_when_fallback_splits_dir_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(explain_cli, "get_output_root", lambda: tmp_path)

    args = argparse.Namespace(
        models=["all"], run=None, baseline=True, output_dir=None, image=None, output=None
    )
    cfg = {"lime": {"baseline": True}, "gradcam": {"enabled": False}}

    with pytest.raises(SystemExit, match="El directorio de splits no existe"):
        explain_cli.cmd_visual(args, cfg, device=None)


def test_load_visual_report_functions_reports_missing_xai_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *import_args, **import_kwargs):
        if name == "src.explainability.visual_report":
            raise ModuleNotFoundError("No module named 'lime'", name="lime")
        return real_import(name, *import_args, **import_kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit, match="Instala el extra xai"):
        explain_cli._load_visual_report_functions()


def test_compare_accepts_shap_overrides():
    args = build_parser().parse_args(["compare", "--sample-size", "3", "--nsamples", "128"])

    assert args.sample_size == 3
    assert args.nsamples == 128


def test_main_dispatches_compare_and_global_to_their_handlers(monkeypatch):
    """`main()` resuelve `compare` y `global` a callables reales en su dict `handlers`,
    no a los stubs que este test reemplaza: se parchean `cmd_compare`/`cmd_global` con
    sentinelas y se verifica que el subcomando invocado llega exactamente a la suya."""
    calls: list[str] = []

    def _fake_handler(name):
        return lambda args, cfg, device: calls.append(name)

    monkeypatch.setattr(explain_cli, "cmd_compare", _fake_handler("compare"))
    monkeypatch.setattr(explain_cli, "cmd_global", _fake_handler("global"))
    monkeypatch.setattr(explain_cli, "load_config", lambda: {"lime": {"seed": 42}})
    monkeypatch.setattr(explain_cli, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(explain_cli, "set_global_seed", lambda seed: None)

    monkeypatch.setattr("sys.argv", ["explain.py", "compare"])
    explain_cli.main()
    monkeypatch.setattr("sys.argv", ["explain.py", "global"])
    explain_cli.main()

    assert calls == ["compare", "global"]


def _dummy_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2)).eval()


def test_cmd_global_skips_cleanly_when_nothing_could_be_accumulated(tmp_path, monkeypatch, caplog):
    """Cubre a nivel de CLI el guard de `cmd_global` (scripts/pipeline/explain.py:682):
    si todas las imagenes de la muestra fallan al cargar, el acumulador queda vacio y el
    comando debe loguear y continuar, no escribir `explain_global/` ni lanzar."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    predictions = pd.DataFrame(
        {
            "image_path": ["falta/uno.png", "falta/dos.png"],
            "label": ["healthy", "common_rust"],
            "pred_label": ["healthy", "healthy"],
            "pred_prob": [0.9, 0.4],
        }
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    context = RunContext(
        model_name="dummy",
        run_dir=run_dir,
        model=_dummy_model(),
        idx_to_class={0: "healthy", 1: "common_rust"},
        target_size=(8, 8),
        splits_dir=tmp_path / "splits",
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(explain_cli, "iter_run_contexts", lambda *args, **kwargs: iter([context]))
    monkeypatch.setattr(explain_cli, "get_dataset_root", lambda: tmp_path / "dataset_inexistente")

    args = argparse.Namespace(
        models=["all"],
        run=None,
        baseline=None,
        output_dir=None,
        sample_size=None,
        nsamples=None,
    )
    cfg = {
        "shap": {
            "segmentation": "slic",
            "n_segments": 4,
            "compactness": 10.0,
            "nsamples": 32,
            "batch_size": 8,
            "background": "black",
            "seed": 42,
            "global_sample_size": 5,
        }
    }

    with caplog.at_level(logging.INFO):
        explain_cli.cmd_global(args, cfg, device=torch.device("cpu"))

    assert not (run_dir / "explain_global").exists()
    assert "Nada que perfilar" in caplog.text
