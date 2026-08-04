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
