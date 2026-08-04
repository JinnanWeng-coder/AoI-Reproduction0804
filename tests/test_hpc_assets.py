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
    assert "/eeedata/sgxyl2/AoI-Reproduction0804-results" in audit


def test_grok_handoff_contains_non_destructive_scientific_gates():
    handoff = _read("GROK_HANDOFF.md")
    for requirement in (
        "Never run `final_test`",
        "Never delete, rename, reset or overwrite a formal run directory",
        "Never write result artifacts into the Git checkout",
        "Do not attempt DDP/DataParallel",
        "Algorithm 2 only",
    ):
        assert requirement in handoff
