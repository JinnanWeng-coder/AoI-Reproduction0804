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
    paths = [
        path
        for pattern in patterns
        for path in sorted((ROOT / "hpc").glob(pattern))
        if "_eval_" not in path.name
    ]
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


def test_mappo_stability_array_has_exactly_two_single_factor_arms():
    text = (ROOT / "hpc" / "aoi_mappo_stability_array.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --array=0-11%6",
        'arm="actor_lr1e4"',
        'arm="entropy2x"',
        'actor_lr="0.0001"',
        'actor_lr="0.0005"',
        'entropy_rb="0.02"',
        'entropy_mode="0.02"',
        'entropy_power="0.002"',
        "seeds=(8 9 10 11 12 13)",
        "MAPPO_results/stability-ablation-v1/P5_N4_gap25",
    ):
        assert required in text
    assert "--eval-only" not in text
    assert "--checkpoint-mode resumable" not in text


def test_mappo_combined_array_has_the_six_cell_confirmation_contract():
    text = (ROOT / "hpc" / "aoi_mappo_combined_array.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --array=0-5%6",
        'arm="actor_lr1e4_entropy2x"',
        'actor_lr="0.0001"',
        'entropy_rb="0.02"',
        'entropy_mode="0.02"',
        'entropy_power="0.002"',
        "seeds=(8 9 10 11 12 13)",
        "MAPPO_results/combined-confirm-v1/P5_N4_gap25",
        "--episodes 500",
    ):
        assert required in text
    assert "--eval-only" not in text
    assert "--checkpoint-mode resumable" not in text


def test_mappo_policy_eval_array_has_the_48_cell_heldout_contract():
    text = (ROOT / "hpc" / "aoi_mappo_policy_eval_array.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --array=0-47%8",
        "arms=(baseline actor_lr1e4 entropy2x actor_lr1e4_entropy2x)",
        "modes=(deterministic stochastic)",
        'EVAL_SEEDS="201,202,203,204,205,206"',
        "MAPPO_results/policy-eval-v1/P5_N4_gap25",
        "MAPPO_results/default/P5_N4_gap25",
        "MAPPO_results/stability-ablation-v1/P5_N4_gap25",
        "MAPPO_results/combined-confirm-v1/P5_N4_gap25",
        "--eval-episodes 100",
        "--diagnostic-eval",
        '--mappo-eval-mode "$mode"',
    ):
        assert required in text
    assert "--checkpoint-mode resumable" not in text
    assert "final_test" not in text


def test_mappo_tdec_ab_arrays_freeze_one_factor_and_requested_resources():
    train = (ROOT / "hpc" / "aoi_mappo_tdec_ab_train_array.sbatch").read_text(encoding="utf-8")
    evaluate = (ROOT / "hpc" / "aoi_mappo_tdec_ab_eval_array.sbatch").read_text(encoding="utf-8")
    for text in (train, evaluate):
        for required in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:l20:1",
            "MAPPO_results/tdec-ab-v1/P5_N4_gap25",
            "variants=(combined tdec)",
            "seeds=(8 9 10 11 12 13)",
            'actor_lr="0.0005"',
            'entropy_rb="0.02"',
            'entropy_mode="0.02"',
            'entropy_power="0.002"',
            'value_clip_mode="normalized"',
            '--mappo-variant "$variant"',
            '--mappo-value-clip-mode "$value_clip_mode"',
        ):
            assert required in text
    assert "#SBATCH --array=0-11%12" in train
    assert "--episodes 500" in train
    assert "--eval-only" not in train
    assert "#SBATCH --array=0-23%12" in evaluate
    assert "modes=(deterministic stochastic)" in evaluate
    assert 'EVAL_SEEDS="201,202,203,204,205,206"' in evaluate
    assert "--eval-episodes 100" in evaluate
    assert "--diagnostic-eval" in evaluate
    assert "final_test" not in train and "final_test" not in evaluate


def test_mappo_gradient_conflict_audit_has_the_diagnostic_only_contract():
    text = (ROOT / "hpc" / "aoi_mappo_gradient_conflict_audit_array.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --gres=gpu:l20:1",
        "#SBATCH --array=0-5%6",
        'SCENARIO="p05_n04_g25"',
        "seeds=(8 9 10 11 12 13)",
        'actor_lr="0.0005"',
        'entropy_rb="0.02"',
        'entropy_mode="0.02"',
        'entropy_power="0.002"',
        'value_clip_mode="normalized"',
        "MAPPO_results/gradient-conflict-audit-v1/P5_N4_gap25",
        "--episodes 500",
        "--checkpoint-mode none",
        "--mappo-variant tdec",
        "--mappo-objective-gradient-diagnostics",
    ):
        assert required in text
    assert "--eval-only" not in text
    assert "final_test" not in text


def test_mappo_gradient_conflict_phase2_arrays_have_exact_contract_and_resources():
    train = (ROOT / "hpc" / "aoi_mappo_gradient_conflict_phase2_train_array.sbatch").read_text(encoding="utf-8")
    evaluate = (ROOT / "hpc" / "aoi_mappo_gradient_conflict_phase2_eval_array.sbatch").read_text(encoding="utf-8")
    for text in (train, evaluate):
        for required in (
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --gres=gpu:l20:1",
            "MAPPO_results/gradient-conflict-phase2-v1/P5_N4_gap25",
            "arms=(composed_clip separate_sum_clip pcgrad)",
            "seeds=(8 9 10 11 12 13)",
            'actor_lr="0.0005"',
            'entropy_rb="0.02"',
            'entropy_mode="0.02"',
            'entropy_power="0.002"',
            'value_clip_mode="normalized"',
            "--mappo-variant tdec",
            '--mappo-actor-update-mode "$arm"',
            "--mappo-objective-gradient-diagnostics",
        ):
            assert required in text
        assert "final_test" not in text
        assert "--checkpoint-mode resumable" not in text
    assert "#SBATCH --array=0-17%6" in train
    assert "--episodes 500" in train
    assert "--checkpoint-mode policy_only" in train
    assert "--eval-only" not in train
    assert "#SBATCH --array=0-35%12" in evaluate
    assert "modes=(deterministic stochastic)" in evaluate
    assert 'EVAL_SEEDS="201,202,203,204,205,206"' in evaluate
    assert "--eval-episodes 100" in evaluate
    assert "--diagnostic-eval" in evaluate
    assert '--mappo-eval-mode "$mode"' in evaluate


def test_phase2_task_mapping_places_seed8_pilots_at_0_6_12():
    arms = ("composed_clip", "separate_sum_clip", "pcgrad")
    seeds = (8, 9, 10, 11, 12, 13)
    mapping = {
        task: (arms[task // 6], seeds[task % 6])
        for task in range(18)
    }
    assert {task for task, (_arm, seed) in mapping.items() if seed == 8} == {0, 6, 12}
    assert len(set(mapping.values())) == 18


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
