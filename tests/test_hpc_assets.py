from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HPC = ROOT / "hpc"


def _read(name: str) -> str:
    return (HPC / name).read_text(encoding="utf-8")


def test_eight_gpu_job_uses_eight_exclusive_single_gpu_shards():
    script = _read("aoi_matrix_8gpu.sbatch")
    assert "#SBATCH -p Q10" in script
    assert "#SBATCH --gres=gpu:l20:8" in script
    assert "SHARD_COUNT=8" in script
    assert "srun --exclusive" in script
    assert "--gres=gpu:l20:1" in script
    assert "--shard-count \"$SHARD_COUNT\"" in script
    assert "--shard-index \"$shard_index\"" in script
    assert "PILOT_APPROVED" in script
    assert "TRAIN_APPROVED" in script
    assert "final_test" not in script


def test_gpu_array_maps_one_formal_cell_to_each_single_gpu_task():
    script = _read("aoi_matrix_array.sbatch")
    assert "#SBATCH -p Q10" in script
    assert "#SBATCH --array=0-47%8" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --gres=gpu:l20:1" in script
    assert "CELL_COUNT=48" in script
    assert '--shard-count "$CELL_COUNT"' in script
    assert '--shard-index "$SLURM_ARRAY_TASK_ID"' in script
    assert "%x_%A_%a.out" in script
    assert "PILOT_APPROVED" in script
    assert "TRAIN_APPROVED" in script
    assert "final_test" not in script


def test_diagnostic_array_is_small_external_and_validation_only():
    script = _read("aoi_diagnostic_array.sbatch")
    for token in (
        "#SBATCH --array=0-8%8",
        "#SBATCH --gres=gpu:l20:1",
        "task_count=9",
        "task_count=36",
        "arms=(current global_detached legacy_release)",
        "seeds=(3 5 7)",
        "noises=(0 0.05 0.1 0.3)",
        "noise_tokens=(0p0 0p05 0p1 0p3)",
        'SCENARIO="p05_n04_g25"',
        "--global-actor-mode synchronous_joint",
        "--global-actor-mode detached_actor",
        "--diagnostics",
        "--eval-purpose validation",
        "--eval-noise",
        "--eval-seeds",
        "AoI-Reproduction-diagnostics/global-causal-v1",
        "SKIP completed training cell",
        "SKIP completed validation",
    ):
        assert token in script
    assert "final_test" not in script
    assert "PILOT_APPROVED" not in script
    assert "TRAIN_APPROVED" not in script


def test_hpc_readme_keeps_diagnostic_and_formal_arrays_distinct():
    readme = _read("README_HPC.md")
    for token in (
        "Small causal diagnostic array (not the formal matrix)",
        "--array=0-8%8",
        "--array=0-35%8",
        "AOI_STAGE=train",
        "AOI_STAGE=eval",
        "selection-validation winner in `best.pt`",
    ):
        assert token in readme


def test_pilot_preserves_exact_formal_identity_and_validation_split():
    script = _read("aoi_pilot_1gpu.sbatch")
    for token in (
        "--profile paper_faithful",
        "--scenario p05_n04_g25",
        "--seed 2",
        "--device cuda:0",
        "--checkpoint-every 5",
        "--eval-purpose validation",
        "--eval-seeds 201,202,203,204,205,206",
        "--recover-empty-run",
    ):
        assert token in script
    assert "#SBATCH --gres=gpu:l20:1" in script
    assert "VALIDATION_READY.json" in script
    assert "final_test" not in script


def test_environment_and_audit_assets_are_pinned_and_external():
    setup = _read("setup_aoi_cuda.sh")
    audit = _read("aoi_audit_cpu.sbatch")
    assert "python=3.10.20" in setup
    assert "torch==2.11.0" in setup
    assert "https://download.pytorch.org/whl/cu126" in setup
    assert "requirements.cuda.lock.txt" in setup
    assert "git status --porcelain --untracked-files=all" in setup
    assert "Expected exactly 48 formal run directories" in audit
    assert "AOI_AUDIT_SCOPE" in audit
    assert "/eeedata/sgxjw2/AoI-Reproduction0804-results" in audit
    for name in (
        "setup_aoi_cuda.sh",
        "aoi_pilot_1gpu.sbatch",
        "aoi_matrix_8gpu.sbatch",
        "aoi_matrix_array.sbatch",
        "aoi_diagnostic_array.sbatch",
        "aoi_audit_cpu.sbatch",
    ):
        text = _read(name)
        assert "sgxyl2" not in text
        assert "/share/home" not in text
        assert "/eeedata/sgxjw2" in text


def test_grok_handoff_contains_non_destructive_scientific_gates():
    handoff = _read("GROK_HANDOFF.md")
    for requirement in (
        "hpc/aoi_matrix_array.sbatch",
        "Never run `final_test`",
        "Never delete, rename, reset or overwrite a formal run directory",
        "Never write result artifacts into the Git checkout",
        "Never run the full-node and array drivers",
        "requires a fresh pilot and fresh",
        "Do not attempt DDP/DataParallel",
        "Algorithm 2 only",
    ):
        assert requirement in handoff
