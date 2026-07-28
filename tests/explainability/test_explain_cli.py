import pytest

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
