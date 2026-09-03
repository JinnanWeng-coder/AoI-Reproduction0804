from pathlib import Path

import pytest

from aoi_v2x_reproduction.config import (
    REPRODUCTION_PROFILE,
    all_scenarios,
    build_parser,
    config_from_dict,
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
        "--mappo-variant", "tdec",
        "--mappo-actor-lr", "0.0001",
        "--mappo-entropy-coef-rb", "0.02",
        "--mappo-entropy-coef-mode", "0.02",
        "--mappo-entropy-coef-power", "0.002",
        "--mappo-value-clip-mode", "legacy_raw",
    ])
    config = config_from_args(args)
    assert config.mappo_variant == "tdec"
    assert config.mappo_actor_lr == pytest.approx(0.0001)
    assert config.mappo_entropy_coef_rb == pytest.approx(0.02)
    assert config.mappo_entropy_coef_mode == pytest.approx(0.02)
    assert config.mappo_entropy_coef_power == pytest.approx(0.002)
    assert config.mappo_value_clip_mode == "legacy_raw"
    assert config.to_dict()["mappo_value_clip_mode"] == "legacy_raw"

    invalid = parser.parse_args(["--algorithm", "modified_maddpg", "--mappo-actor-lr", "0.0001"])
    with pytest.raises(ValueError, match="require --algorithm mappo"):
        config_from_args(invalid)


def test_cli_exposes_tdec_objective_gradient_diagnostics_and_defaults_off():
    default = resolve_config(scenario="p05_n04_g25", algorithm="mappo", mappo_variant="tdec")
    assert default.mappo_objective_gradient_diagnostics is False
    assert default.to_dict()["mappo_objective_gradient_diagnostics"] is False

    args = build_parser().parse_args([
        "--algorithm", "mappo",
        "--mappo-variant", "tdec",
        "--mappo-objective-gradient-diagnostics",
    ])
    enabled = config_from_args(args)
    assert enabled.mappo_objective_gradient_diagnostics is True
    assert enabled.to_dict()["mappo_objective_gradient_diagnostics"] is True

    invalid_variant = build_parser().parse_args([
        "--algorithm", "mappo",
        "--mappo-objective-gradient-diagnostics",
    ])
    with pytest.raises(ValueError, match="requires algorithm=mappo and mappo_variant=tdec"):
        config_from_args(invalid_variant)


def test_cli_exposes_phase2_actor_update_modes_and_restricts_objective_modes_to_tdec():
    default = resolve_config(scenario="p05_n04_g25", algorithm="mappo", mappo_variant="tdec")
    assert default.mappo_actor_update_mode == "composed_clip"

    for mode in ("composed_clip", "separate_sum_clip", "pcgrad"):
        args = build_parser().parse_args([
            "--algorithm", "mappo",
            "--mappo-variant", "tdec",
            "--mappo-actor-update-mode", mode,
        ])
        assert config_from_args(args).mappo_actor_update_mode == mode

    invalid = build_parser().parse_args([
        "--algorithm", "mappo",
        "--mappo-variant", "combined",
        "--mappo-actor-update-mode", "pcgrad",
    ])
    with pytest.raises(ValueError, match="objective-wise MAPPO actor updates"):
        config_from_args(invalid)


def test_missing_actor_update_mode_preserves_historical_mappo_identity():
    current = resolve_config(scenario="p05_n04_g25", algorithm="mappo", mappo_variant="tdec")
    historical = current.to_dict()
    historical.pop("mappo_actor_update_mode")
    reconstructed = config_from_dict(historical)
    assert reconstructed.mappo_actor_update_mode == "composed_clip"
    assert reconstructed.to_dict() == historical


def test_mappo_value_clipping_defaults_to_normalized_and_unversioned_configs_remain_legacy():
    current = resolve_config(scenario="p05_n04_g25", algorithm="mappo")
    assert current.mappo_value_clip_mode == "normalized"
    assert current.to_dict()["mappo_value_clip_mode"] == "normalized"

    unversioned = current.to_dict()
    unversioned.pop("mappo_value_clip_mode")
    reconstructed = config_from_dict(unversioned)
    assert reconstructed.mappo_value_clip_mode == "legacy_raw"
    assert reconstructed.to_dict() == unversioned


def test_mappo_variant_defaults_to_combined_and_missing_field_preserves_historical_identity():
    current = resolve_config(scenario="p05_n04_g25", algorithm="mappo")
    assert current.mappo_variant == "combined"
    assert current.to_dict()["mappo_variant"] == "combined"

    historical = current.to_dict()
    historical.pop("mappo_variant")
    reconstructed = config_from_dict(historical)
    assert reconstructed.mappo_variant == "combined"
    assert reconstructed.to_dict() == historical

    original_mappo = dict(historical)
    original_mappo.pop("mappo_value_clip_mode")
    original_reconstructed = config_from_dict(original_mappo)
    assert original_reconstructed.mappo_variant == "combined"
    assert original_reconstructed.mappo_value_clip_mode == "legacy_raw"
    assert original_reconstructed.to_dict() == original_mappo

    tdec = resolve_config(scenario="p05_n04_g25", algorithm="mappo", mappo_variant="tdec")
    assert tdec.canonical_hash() != current.canonical_hash()


def test_missing_objective_gradient_field_preserves_historical_mappo_identity():
    current = resolve_config(scenario="p05_n04_g25", algorithm="mappo", mappo_variant="tdec")
    historical = current.to_dict()
    historical.pop("mappo_objective_gradient_diagnostics")
    reconstructed = config_from_dict(historical)
    assert reconstructed.mappo_objective_gradient_diagnostics is False
    assert reconstructed.to_dict() == historical


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
