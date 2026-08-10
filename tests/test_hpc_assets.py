from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_training_arrays_do_not_request_resumable_checkpoints():
    for path in sorted((ROOT / "hpc").glob("aoi_modified_maddpg*_array.sbatch")):
        text = path.read_text(encoding="utf-8")
        assert "--checkpoint-mode resumable" not in text
        assert "--eval-only" not in text
