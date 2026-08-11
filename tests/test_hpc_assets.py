from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_PARENT = "/eeedata/sgxjw2/Parvini-TVT2023-reproduction"


def test_active_hpc_scripts_use_the_single_baseline_and_new_package():
    scripts = sorted((ROOT / "hpc").glob("*.sbatch"))
    assert scripts
    combined = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
    for removed in (
        "--profile paper_faithful",
        "--tau ",
        "--slow-update-every-episodes",
        "--global-actor-mode",
        "from config import",
        "SOURCE_MANIFEST",
        "legacy_release",
    ):
        assert removed not in combined
    assert "aoi_v2x_reproduction" in combined
    assert REMOTE_PARENT in combined
    assert "/eeedata/sgxjw2/AoI-Reproduction0804-results" not in combined
    assert "/eeedata/sgxjw2/AoI-Reproduction-diagnostics" not in combined


def test_training_arrays_do_not_request_resumable_checkpoints():
    patterns = ("aoi_modified_maddpg*_array.sbatch", "aoi_mappo*_array.sbatch")
    paths = [path for pattern in patterns for path in sorted((ROOT / "hpc").glob(pattern))]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "--checkpoint-mode resumable" not in text
        assert "--eval-only" not in text


def test_mappo_default_array_has_the_frozen_first_wave_contract():
    text = (ROOT / "hpc" / "aoi_mappo_default_array.sbatch").read_text(encoding="utf-8")
    for required in (
        'ALGORITHM="mappo"',
        'SCENARIO="p05_n04_g25"',
        "seeds=(8 9 10 11 12 13)",
        "--episodes 500",
        "mappo_rollout_episodes == 5",
        "mappo_ppo_epochs == 10",
        "MAPPO_results/default/P5_N4_gap25",
        "learning_diagnostics.json",
    ):
        assert required in text
    assert "eval-only" not in text
    assert "formal" not in text.lower()


def test_deferred_heldout_scripts_require_the_dedicated_result_root():
    for name in (
        "aoi_pilot_1gpu.sbatch",
        "aoi_matrix_8gpu.sbatch",
        "aoi_matrix_array.sbatch",
        "aoi_audit_cpu.sbatch",
    ):
        text = (ROOT / "hpc" / name).read_text(encoding="utf-8")
        assert '${AOI_RESULT_ROOT:?' in text
        assert "Modified_MADDPG_with_TDec_results/heldout-formal-matrix" in text
        assert 'RESULT_ROOT="${AOI_RESULT_ROOT:-' not in text
