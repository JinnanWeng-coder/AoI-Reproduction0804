from pathlib import Path

import pytest

from aoi_v2x_reproduction.config import (
    REPRODUCTION_PROFILE,
    all_scenarios,
    build_parser,
    config_from_args,
    matrix_specs,
    resolve_config,
    safe_run_dir,
)


def test_reproduction_baseline_is_the_only_active_profile():
    config = resolve_config(scenario="p05_n04_g25")
    assert REPRODUCTION_PROFILE == "reproduction_baseline"
    assert config.profile == REPRODUCTION_PROFILE
    assert config.semantic_version == "reproduction_baseline_v1"
    assert config.tau == pytest.approx(0.005)
    assert config.global_actor_weight == pytest.approx(1.0)
    assert config.global_update_mode == "synchronous_joint"
    assert config.slow_update_every_episodes == 1
    assert config.checkpoint_mode == "policy_only"
    assert config.state_dim == 22
    assert config.is_formal_result is False
    with pytest.raises(ValueError, match="profile must be one of"):
        resolve_config("paper_faithful", "p05_n04_g25")


def test_baseline_knobs_are_locked():
    with pytest.raises(ValueError, match="tau"):
        resolve_config(scenario="p05_n04_g25", tau=0.0005)
    with pytest.raises(ValueError, match="slow_update_every_episodes"):
        resolve_config(scenario="p05_n04_g25", slow_update_every_episodes=20)
    with pytest.raises(ValueError, match="unsupported global_update_mode"):
        resolve_config(scenario="p05_n04_g25", global_update_mode="sequential_agent")


def test_matrix_is_exactly_48_unique_tasks():
    specs = matrix_specs()
    keys = {(item["profile"], item["scenario"], item["seed"]) for item in specs}
    assert len(all_scenarios()) == 8
    assert len(specs) == 48
    assert len(keys) == 48
    assert {item["seed"] for item in specs} == set(range(2, 8))
    assert {item["profile"] for item in specs} == {REPRODUCTION_PROFILE}


def test_safe_run_path_rejects_escape():
    root = Path("scratch")
    assert safe_run_dir(root, "valid-run").parent.name == "scratch"
    with pytest.raises(ValueError):
        safe_run_dir(root, "../escape")
    with pytest.raises(ValueError):
        safe_run_dir(root, "nested/name")


def test_cli_exposes_artifact_policy_but_not_baseline_knobs():
    parser = build_parser()
    args = parser.parse_args(["--diagnostics", "--checkpoint-mode", "none"])
    config = config_from_args(args)
    assert config.diagnostics is True
    assert config.checkpoint_mode == "none"
    assert config.tau == pytest.approx(0.005)
    with pytest.raises(SystemExit):
        parser.parse_args(["--tau", "0.0005"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--profile", "paper_faithful"])


def test_cli_exposes_only_the_planned_mappo_stability_overrides():
    parser = build_parser()
    args = parser.parse_args([
        "--algorithm", "mappo",
        "--mappo-actor-lr", "0.0001",
        "--mappo-entropy-coef-rb", "0.02",
        "--mappo-entropy-coef-mode", "0.02",
        "--mappo-entropy-coef-power", "0.002",
    ])
    config = config_from_args(args)
    assert config.mappo_actor_lr == pytest.approx(0.0001)
    assert config.mappo_entropy_coef_rb == pytest.approx(0.02)
    assert config.mappo_entropy_coef_mode == pytest.approx(0.02)
    assert config.mappo_entropy_coef_power == pytest.approx(0.002)

    invalid = parser.parse_args(["--algorithm", "modified_maddpg", "--mappo-actor-lr", "0.0001"])
    with pytest.raises(ValueError, match="require --algorithm mappo"):
        config_from_args(invalid)


def test_diagnostic_eval_flag_is_not_valid_for_training():
    from Main import main

    with pytest.raises(SystemExit, match="only valid with --eval-only"):
        main(["--diagnostic-eval", "--dry-run"])


def test_cli_parses_explicit_mappo_policy_eval_mode():
    args = build_parser().parse_args([
        "--algorithm", "mappo",
        "--eval-only",
        "--diagnostic-eval",
        "--mappo-eval-mode", "deterministic",
    ])
    assert args.mappo_eval_mode == "deterministic"
