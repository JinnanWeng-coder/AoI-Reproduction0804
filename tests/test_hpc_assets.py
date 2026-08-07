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


def test_tau_slow_array_is_the_bounded_two_by_two_recovery_experiment():
    script = _read("aoi_tau_slow_array.sbatch")
    for token in (
        "#SBATCH --array=0-11%8",
        "task_count=12",
        "task_count=48",
        "taus=(0.0005 0.005)",
        "slow_intervals=(1 20)",
        "seeds=(3 5 7)",
        "checkpoint_names=(best.pt latest.pt)",
        "noises=(0 0.3)",
        '--profile paper_faithful',
        '--global-actor-mode synchronous_joint',
        '--tau "$tau"',
        '--slow-update-every-episodes "$slow_interval"',
        "--diagnostics",
        "--diagnostic-eval",
        "--eval-purpose validation",
        "--eval-seeds",
        "AoI-Reproduction-diagnostics/tau-slow-v1",
    ):
        assert token in script
    assert "legacy_release" not in script
    assert "detached_actor" not in script
    assert "final_test" not in script


def test_hpc_readme_documents_tau_slow_dependency_and_two_checkpoints():
    readme = _read("README_HPC.md")
    for token in (
        "Tau x slow-update recovery diagnostic",
        "--array=0-11%8",
        "--array=0-47%8",
        'dependency=afterok:',
        "`best.pt`",
        "`latest.pt`",
        "201..206",
    ):
        assert token in readme


def test_tau005_confirmation_array_is_exactly_six_train_and_twenty_four_eval_tasks():
    script = _read("aoi_tau005_confirm_array.sbatch")
    for token in (
        "#SBATCH --array=0-5%6",
        "task_count=6",
        "task_count=24",
        "seeds=(2 3 4 5 6 7)",
        'TAU="0.005"',
        'SLOW_INTERVAL="1"',
        'SCENARIO="p05_n04_g25"',
        "checkpoint_names=(best.pt latest.pt)",
        "noises=(0 0.3)",
        "--profile paper_faithful",
        "--episodes 500",
        '--tau "$TAU"',
        '--slow-update-every-episodes "$SLOW_INTERVAL"',
        "--global-actor-mode synchronous_joint",
        "--diagnostics",
        "--diagnostic-eval",
        "--eval-episodes 100",
        "--eval-seeds",
        "--eval-noise",
        "AoI-Reproduction-diagnostics/tau005-confirm-v1",
    ):
        assert token in script
    for forbidden in ("legacy_release", "detached_actor", "final_test", "noise-decay", "600"):
        assert forbidden not in script
    assert "PILOT_APPROVED" not in script
    assert "TRAIN_APPROVED" not in script


def test_hpc_readme_documents_tau005_confirmation_submission():
    readme = _read("README_HPC.md")
    for token in (
        "Tau 0.005 six-seed confirmation",
        "aoi_tau005_confirm_array.sbatch",
        "--array=0-5%6",
        "--array=0-23%8",
        'dependency=afterok:',
        "fixed training noise 0.3",
        "201..206",
    ):
        assert token in readme


def test_gap_trend_array_adds_only_two_missing_gap_anchors():
    script = _read("aoi_gap_trend_array.sbatch")
    for token in (
        "#SBATCH --array=0-11%8",
        "task_count=12",
        "task_count=24",
        "scenarios=(p05_n04_g05 p05_n04_g35)",
        "seeds=(2 3 4 5 6 7)",
        "checkpoint_names=(best.pt latest.pt)",
        'TAU="0.005"',
        'SLOW_INTERVAL="1"',
        'TRAIN_NOISE="0.3"',
        'EVAL_NOISE="0.3"',
        "--profile paper_faithful",
        "--episodes 500",
        '--tau "$TAU"',
        '--slow-update-every-episodes "$SLOW_INTERVAL"',
        "--global-actor-mode synchronous_joint",
        "--diagnostics",
        "--diagnostic-eval",
        "--eval-episodes 100",
        "--eval-seeds",
        '--eval-noise "$EVAL_NOISE"',
        "AoI-Reproduction-diagnostics/gap-trend-v1",
    ):
        assert token in script
    for forbidden in (
        "p05_n04_g25",
        "legacy_release",
        "detached_actor",
        "final_test",
        "noise-decay",
        "600",
        "PILOT_APPROVED",
        "TRAIN_APPROVED",
    ):
        assert forbidden not in script


def test_hpc_readme_documents_gap_trend_dependency_and_reused_anchor():
    readme = _read("README_HPC.md")
    for token in (
        "Tau 0.005 gap-trend pilot",
        "aoi_gap_trend_array.sbatch",
        "p05_n04_g05",
        "p05_n04_g35",
        "tau005-confirm-v1",
        "--array=0-11%8",
        "--array=0-23%8",
        'dependency=afterok:',
        "best.pt",
        "latest.pt",
        "noise 0.3",
        "binary endpoint CAM",
        "payload completion",
    ):
        assert token in readme


def test_gap15_fill_array_is_six_training_cells_only():
    script = _read("aoi_gap15_fill_array.sbatch")
    for token in (
        "#SBATCH --array=0-5%6",
        "seeds=(2 3 4 5 6 7)",
        'SCENARIO="p05_n04_g15"',
        'TAU="0.005"',
        'SLOW_INTERVAL="1"',
        'TRAIN_NOISE="0.3"',
        "--profile paper_faithful",
        "--episodes 500",
        '--tau "$TAU"',
        '--slow-update-every-episodes "$SLOW_INTERVAL"',
        "--global-actor-mode synchronous_joint",
        "--diagnostics",
        "--scope train",
        "primary_metric_window=last100",
        "AoI-Reproduction-diagnostics/gap15-fill-v1",
    ):
        assert token in script
    for forbidden in (
        "AOI_STAGE",
        "--eval-only",
        "--eval-purpose",
        "--eval-seeds",
        "--eval-noise",
        "p05_n04_g05",
        "p05_n04_g25",
        "p05_n04_g35",
        "final_test",
        "PILOT_APPROVED",
        "TRAIN_APPROVED",
    ):
        assert forbidden not in script


def test_hpc_readme_documents_gap15_as_train_last100_only():
    readme = _read("README_HPC.md")
    for token in (
        "Gap 15 source-protocol fill",
        "aoi_gap15_fill_array.sbatch",
        "p05_n04_g15",
        "--array=0-5%6",
        "final 100 training episodes",
        "held-out eval array",
        "strict binary endpoint CAM",
        "continuous endpoint payload completion",
    ):
        assert token in readme


def test_gap_global_slow_array_is_exactly_24_plus_18_training_cells():
    script = _read("aoi_gap_global_slow_42_array.sbatch")
    for token in (
        "#SBATCH --array=0-23%8",
        'PHASE="${AOI_PHASE:-A}"',
        "PHASE_A_CELL_COUNT=24",
        "PHASE_B_CELL_COUNT=18",
        "TOTAL_UNIQUE_TRAINING_CELLS=42",
        "seeds=(8 9 10 11 12 13)",
        "phase_a_scenarios=(p05_n04_g05 p05_n04_g15 p05_n04_g25 p05_n04_g35)",
        "phase_b_arm_labels=(sync_slow20 detached_slow01 detached_slow20)",
        "phase_b_slow_intervals=(20 1 20)",
        "phase_b_global_modes=(synchronous_joint detached_actor detached_actor)",
        'TAU="0.005"',
        'TRAIN_NOISE="0.3"',
        "--profile paper_faithful",
        "--episodes 500",
        '--tau "$TAU"',
        '--slow-update-every-episodes "$slow_interval"',
        '--global-actor-mode "$global_mode"',
        "--diagnostics",
        "--scope train",
        "primary_metric_window=last100",
        "secondary_metric_window=last50",
        "AoI-Reproduction-diagnostics/gap-global-slow-42-v1",
    ):
        assert token in script
    for forbidden in (
        "--eval-only",
        "--eval-purpose",
        "--eval-seeds",
        "--eval-noise",
        "legacy_release",
        "final_test",
        "PILOT_APPROVED",
        "TRAIN_APPROVED",
    ):
        assert forbidden not in script


def test_hpc_readme_documents_gap_global_slow_42_cell_submission():
    readme = _read("README_HPC.md")
    for token in (
        "Seeds 8--13 gap and gap-25 mechanism check (42 training cells)",
        "aoi_gap_global_slow_42_array.sbatch",
        "4 gaps x 6 seeds = 24",
        "3 arms x 6 seeds = 18",
        "24 + 18 = 42",
        "--array=0-23%8",
        "--array=0-17%8",
        "AOI_PHASE=A",
        "AOI_PHASE=B",
        'dependency=afterok:',
        "final 100 episodes",
        "final 50",
        "gap 25 only",
    ):
        assert token in readme


def test_platoon_size_trend_array_adds_exactly_eighteen_training_cells():
    script = _read("aoi_platoon_size_trend_array.sbatch")
    for token in (
        "#SBATCH --array=0-17%8",
        "TASK_COUNT=18",
        "scenarios=(p05_n06_g25 p05_n08_g25 p05_n10_g25)",
        "seeds=(8 9 10 11 12 13)",
        'TAU="0.005"',
        'SLOW_INTERVAL="1"',
        'GLOBAL_MODE="synchronous_joint"',
        'TRAIN_NOISE="0.3"',
        "--profile paper_faithful",
        "--episodes 500",
        '--tau "$TAU"',
        '--slow-update-every-episodes "$SLOW_INTERVAL"',
        '--global-actor-mode "$GLOBAL_MODE"',
        "--diagnostics",
        "--scope train",
        "primary_metric_window=last100",
        "secondary_metric_window=last50",
        "AoI-Reproduction-diagnostics/platoon-size-trend-v1",
    ):
        assert token in script
    for forbidden in (
        "p05_n04_g25",
        "--eval-only",
        "--eval-purpose",
        "--eval-seeds",
        "--eval-noise",
        "detached_actor",
        "legacy_release",
        "final_test",
        "PILOT_APPROVED",
        "TRAIN_APPROVED",
    ):
        assert forbidden not in script


def test_hpc_readme_documents_platoon_size_trend_and_reused_baseline():
    readme = _read("README_HPC.md")
    for token in (
        "Fig. 5(c)/(d) platoon-size trend (18 new training cells)",
        "aoi_platoon_size_trend_array.sbatch",
        "platoon sizes 6, 8, and 10",
        "gap-global-slow-42-v1",
        "3 new platoon sizes x 6 seeds = 18",
        "--array=0-17%8",
        "final 100 episodes",
        "final 50",
        "paired independent unit",
        "strict binary",
        "continuous payload",
        "favorable seed subset",
    ):
        assert token in readme


def test_modified_maddpg_default_array_is_exactly_six_algorithm1_training_cells():
    script = _read("aoi_modified_maddpg_default_array.sbatch")
    for token in (
        "#SBATCH --array=0-5%6",
        "TASK_COUNT=6",
        "seeds=(8 9 10 11 12 13)",
        'ALGORITHM="modified_maddpg"',
        'SCENARIO="p05_n04_g25"',
        'TAU="0.005"',
        'SLOW_INTERVAL="1"',
        'GLOBAL_MODE="synchronous_joint"',
        'TRAIN_NOISE="0.3"',
        "Modified_MADDPG_results/default/P5_N4_gap25",
        '--algorithm "$ALGORITHM"',
        "--profile paper_faithful",
        "--episodes 500",
        '--tau "$TAU"',
        '--slow-update-every-episodes "$SLOW_INTERVAL"',
        '--global-actor-mode "$GLOBAL_MODE"',
        "--diagnostics",
        "--scope train",
        "primary_metric_window=last100",
        "secondary_metric_window=last50",
    ):
        assert token in script
    for forbidden in (
        "--eval-only",
        "--eval-purpose",
        "--eval-seeds",
        "--eval-noise",
        "detached_actor",
        "legacy_release",
        "final_test",
        "PILOT_APPROVED",
        "TRAIN_APPROVED",
    ):
        assert forbidden not in script


def test_hpc_readme_documents_modified_maddpg_default_and_future_subdirectories():
    readme = _read("README_HPC.md")
    for token in (
        "Modified MADDPG Algorithm 1 default confirmation",
        "aoi_modified_maddpg_default_array.sbatch",
        "Modified_MADDPG_results/default/P5_N4_gap25",
        "seeds 8..13",
        "final 100 training episodes",
        "final 50",
        "summarize_modified_maddpg_default.py",
        "gap-extension/",
        "platoon-size-extension/",
        "no eval array",
    ):
        assert token in readme


def test_modified_maddpg_gap_array_is_exactly_eighteen_training_cells():
    script = _read("aoi_modified_maddpg_gap_array.sbatch")
    for token in (
        "#SBATCH --array=0-17%6",
        "TASK_COUNT=18",
        "scenarios=(p05_n04_g05 p05_n04_g15 p05_n04_g35)",
        "seeds=(8 9 10 11 12 13)",
        'ALGORITHM="modified_maddpg"',
        'TAU="0.005"',
        'SLOW_INTERVAL="1"',
        'GLOBAL_MODE="synchronous_joint"',
        'TRAIN_NOISE="0.3"',
        "Modified_MADDPG_results/gap-extension",
        'scenario_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))',
        'seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))',
        'run_name="modified_maddpg_gap_${scenario}_seed${seed_token}"',
        '--algorithm "$ALGORITHM"',
        "--profile paper_faithful",
        '--scenario "$scenario"',
        "--episodes 500",
        '--tau "$TAU"',
        '--slow-update-every-episodes "$SLOW_INTERVAL"',
        '--global-actor-mode "$GLOBAL_MODE"',
        "--diagnostics",
        "--scope train",
        "primary_metric_window=last100",
        "secondary_metric_window=last50",
    ):
        assert token in script
    assert "p05_n04_g25" not in script
    for forbidden in (
        "--eval-only",
        "--eval-purpose",
        "--eval-seeds",
        "--eval-noise",
        "detached_actor",
        "legacy_release",
        "final_test",
        "PILOT_APPROVED",
        "TRAIN_APPROVED",
    ):
        assert forbidden not in script


def test_hpc_readme_documents_modified_maddpg_gap_extension_and_default_reuse():
    readme = _read("README_HPC.md")
    for token in (
        "Modified MADDPG Algorithm 1 gap extension",
        "aoi_modified_maddpg_gap_array.sbatch",
        "exactly 18 cells",
        "gaps 5, 15, and 35 m",
        "seeds 8..13",
        "/eeedata/sgxjw2/Modified_MADDPG_results/gap-extension",
        "summarize_modified_maddpg_gap_extension.py",
        "default_root",
        "not rerun",
        "training-only diagnostic",
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
