import subprocess

from src.config import PROJECT_ROOT


def dry_make(*args):
    return subprocess.check_output(["make", "-n", *args], cwd=PROJECT_ROOT, text=True)


def test_predict_respects_explicit_model_and_run():
    cmd = dry_make("predict", "MODEL=shufflenet_v2_x1_0", "RUN=chosen", "IMAGE=leaf.jpg")
    assert "--model shufflenet_v2_x1_0" in cmd
    assert '--run "chosen"' in cmd
    assert "--model ensemble" in dry_make("predict", "IMAGE=leaf.jpg")


def test_modal_hpo_false_flag_and_split_are_forwarded_without_launching():
    cmd = dry_make(
        "modal-train-main",
        "MAIN_MODELS=efficientnet_b0",
        "BEST_PARAMS=/p.json",
        "CLAHE=0",
        "EVALUATE_TEST=1",
        "SPLITS_DIR=/splits/v2",
    )
    assert '--best-params "/p.json"' in cmd
    assert "--no-clahe" in cmd and "--clahe " not in cmd
    assert '--splits-dir "/splits/v2"' in cmd
    assert "--evaluate-test" in cmd
