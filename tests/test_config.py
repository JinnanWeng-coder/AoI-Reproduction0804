from pathlib import Path

import pytest

from config import all_scenarios, matrix_specs, resolve_config, safe_run_dir


def test_profile_defaults_and_dimensions():
    paper = resolve_config("paper_faithful", "p05_n04_g25")
    legacy = resolve_config("legacy_release", "p05_n04_g25")
    assert paper.tau == pytest.approx(0.0005)
    assert paper.global_actor_weight == pytest.approx(1.0)
    assert paper.global_update_mode == "synchronous_joint"
    assert paper.state_dim == 22
    assert legacy.tau == pytest.approx(0.005)
    assert legacy.global_actor_weight == pytest.approx(2.0)
    assert legacy.global_update_mode == "legacy_detach"
    assert legacy.state_dim == 19


def test_matrix_is_exactly_48_unique_tasks():
    specs = matrix_specs()
    keys = {(item["profile"], item["scenario"], item["seed"]) for item in specs}
    assert len(all_scenarios()) == 8
    assert len(specs) == 48
    assert len(keys) == 48
    assert {item["seed"] for item in specs} == set(range(2, 8))


def test_safe_run_path_rejects_escape():
    root = Path("scratch")
    assert safe_run_dir(root, "valid-run").parent.name == "scratch"
    with pytest.raises(ValueError):
        safe_run_dir(root, "../escape")
    with pytest.raises(ValueError):
        safe_run_dir(root, "nested/name")


def test_unimplemented_sequential_actor_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported global_update_mode"):
        resolve_config("paper_faithful", "p05_n04_g25", global_update_mode="sequential_agent")


def test_detached_actor_is_explicit_nonformal_arm_with_disjoint_selection_split():
    current = resolve_config("paper_faithful", "p05_n04_g25")
    detached = resolve_config("paper_faithful", "p05_n04_g25", global_update_mode="detached_actor")
    assert current.global_update_mode == "synchronous_joint"
    assert current.diagnostics is False
    assert detached.global_update_mode == "detached_actor"
    assert detached.is_formal_result is False
    assert set(current.selection_validation_seeds).isdisjoint(range(101, 207))


def test_diagnostics_and_eval_noise_cli_are_default_off():
    from config import build_parser, config_from_args

    parser = build_parser()
    args = parser.parse_args([])
    config = config_from_args(args)
    assert config.diagnostics is False
    assert args.eval_noise == 0.0
    assert args.diagnostic_eval is False
    args = parser.parse_args([
        "--global-actor-mode", "detached_actor",
        "--diagnostics",
        "--eval-noise", "0.3",
        "--diagnostic-eval",
        "--tau", "0.005",
        "--slow-update-every-episodes", "20",
    ])
    config = config_from_args(args)
    assert config.global_update_mode == "detached_actor"
    assert config.diagnostics is True
    assert config.tau == pytest.approx(0.005)
    assert config.slow_update_every_episodes == 20
    assert args.eval_noise == 0.3
    assert args.diagnostic_eval is True


@pytest.mark.parametrize("tau", [0.0, -0.001, 1.01, float("inf")])
def test_invalid_tau_is_rejected(tau):
    with pytest.raises(ValueError, match="tau"):
        resolve_config("paper_faithful", "p05_n04_g25", tau=tau)


def test_diagnostic_eval_flag_is_not_valid_for_training():
    from Main import main

    with pytest.raises(SystemExit, match="only valid with --eval-only"):
        main(["--diagnostic-eval", "--dry-run"])
