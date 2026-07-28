import argparse
import builtins
import inspect

import pytest

import scripts.pipeline.explain as explain_cli
from scripts.pipeline.explain import build_parser

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


def test_compare_and_global_do_not_raise_the_stub_sentinel():
    from scripts.pipeline import explain

    assert "aun no esta implementado" not in inspect.getsource(explain.cmd_compare)
    assert "aun no esta implementado" not in inspect.getsource(explain.cmd_global)
