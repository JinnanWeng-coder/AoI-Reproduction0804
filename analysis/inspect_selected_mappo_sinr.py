"""Read-only SINR diagnostic of the previously selected lab Shared-TDec best100."""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from analysis import deployment_contract as dc
from analysis import evaluate_training100_selected as selected

BASELINE = "fd99f8810a834e8976eee1fd566159a4975d9dac"
ROOT = Path("SINR-diagnostic/P5_N4_gap25")
TRAINING_COMMIT = "04606e15f26f316728430e53a0e0d1e225bdb3e2"


def code_check(expected):
    commit = dc.git("rev-parse", "HEAD")
    dc.require(commit == expected and dc.git("branch", "--show-current") in {"", "research/mappo"},
               "pinned research checkout required")
    dc.require(not dc.git("status", "--porcelain", "--untracked-files=all"), "clean checkout required")
    dc.require(not dc.git("diff", "--name-only", BASELINE, "HEAD", "--", "aoi_v2x_reproduction"),
               "unexpected algorithm/environment/config change since SINR baseline")
    return commit


def choose(training_batch):
    """Recompute the original training-only choice; never select using test SINR."""
    source = next(s for s in selected.select(selected.candidates(training_batch))
                  if s["source_id"] == "shared_tdec__best100")
    r = source["record"]
    dc.require(r["training_seed"] == 13 and r["best_episode"] == 2109 and r["best_policy_version"] == 421
               and r["implementation_commit"] == TRAINING_COMMIT,
               "selected model differs from the previously accepted seed13/episode2109 best100")
    run = Path(r["run_dir"]).resolve()
    dc.require(run.is_relative_to(training_batch / "training"), "source outside accepted training batch")
    dc.require(Path(r["best"]["path"]).resolve() == run / "policy_best_training.pt", "wrong best100 path")
    complete = dc.read(run / "COMPLETE.json")
    source.update(checkpoint=r["best"], config_hash=complete["config_hash"],
                  policy_episode=r["best_episode"], policy_version=r["best_policy_version"],
                  record_file=dc.file_identity(training_batch / f"records/train_{r['run_id']:02d}.json"),
                  metadata_files=[dc.file_identity(run / n) for n in
                                  ("config.resolved.json", "COMPLETE.json", "training_reward_selection.csv")])
    selected.source_check(source)
    return source


def run(training_batch, batch_id, expected_commit, worlds=(1001,), console_episodes=1, dry_run=False, device="cuda:0"):
    commit = code_check(expected_commit)
    training_batch = Path(training_batch).resolve()
    original = dc.read(training_batch / "protocol/manifest.json")
    dc.require(original["batch_root"] == str(training_batch), "use original lab training batch, not relocated evidence")
    study = Path(original["study_root"]).resolve()
    dc.require(training_batch.parent == study / selected.tc.ROOT, "training batch outside original study root")
    dc.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    worlds = list(worlds)
    dc.require(bool(worlds) and len(set(worlds)) == len(worlds) and set(worlds) <= set(selected.WORLDS),
               "unique, previously used worlds1001-1030 only")
    dc.require(type(console_episodes) is int and 0 <= console_episodes <= 100, "console episodes must be0-100")
    dc.require(device in {"cpu", "cuda:0"}, "device must be cpu or cuda:0")
    out = study / ROOT / batch_id
    dc.require(not out.exists(), "new unique SINR diagnostic directory required; failed attempts must be kept")
    source = choose(training_batch)
    manifest = {"schema": "selected_mappo_sinr_v1", "implementation_commit": commit,
                "source": source, "worlds": worlds, "world_role": selected.WORLD_ROLE,
                "protocol": {"warmup": 5, "scored": 100, "slots": 100, "action_repeat": 0,
                             "execution": "stochastic", "device": device},
                "output_root": str(out), "new_unseen_confirmation": False,
                "note": "Additional SINR diagnostic, not extra independent training evidence or a formal new-world run"}
    if dry_run:
        return {"status": "DRY_RUN", "payload_loaded": False, "rollouts": 0, **manifest}
    # Strict existing loader validates the actual payload BEFORE creating any output.
    selected.source_check(source, payload=True)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "manifest.json", manifest)
    from analysis.evaluate_mappo_zero_shot import evaluate_zero_shot
    result = evaluate_zero_shot(Path(source["checkpoint"]["path"]), out, dc.SCENARIO, device, worlds, 100,
                               deployment=True, world_role=selected.WORLD_ROLE, training_policy=True,
                               sinr_log=True, sinr_console_episodes=console_episodes)
    selected.source_check(source)  # Original checkpoint/config/record identities must remain unchanged.
    native = Path(result["eval_dir"])
    dc.require(result["status"] == "complete" and result["policy_parameters_unchanged"] and
               result["sinr_summary"]["technical_status"] == "PASS" and result["policy_episode"] == 2109 and
               result["training_seed"] == 13 and result["checkpoint_role"] == "best_training100" and
               not result["critic_constructed"] and not result["optimizer_constructed"], "SINR diagnostic validation failed")
    # Detect checkout changes during execution too. No updating a running checkout.
    dc.require(code_check(expected_commit) == commit, "implementation changed during diagnostic")
    verification = {"technical_status": "PASS", "implementation_commit": commit, "worlds": worlds,
                    "training_seed": 13, "policy_episode": 2109, "policy_version": 421,
                    "source_unchanged": True, "row_counts": result["sinr_summary"]["row_counts"],
                    "native_eval_complete": str(native / "EVAL_COMPLETE.json"),
                    "local_trajectory_equivalence_test": "see repository tests; formal reference not rerun"}
    write(out / "verification.json", verification)
    (out / "README.txt").write_text(
        "Shared MAPPO-TDec seed13 best100 episode2109, stochastic repeat0,5warmup+100scored.\n"
        "SINR=S/(I+N), linear ratio and10log10 ratio(dB); signal/interference+noise=mW.\n"
        "Measured using executed-slot channels BEFORE next-slot fading renewal.\n"
        "Inactive mode is null/NA, not0dB. V2V includes3receivers and worst-link SINR.\n"
        "phase separates warmup/scored; summary includes scored only. Averages are link-slot weighted.\n"
        "mean_db is mean of dB, NOT10log10(mean_linear). Values are not CAM deadline success.\n"
        "Episode/slot1-based; platoon/RB0-based; follower1..3. Full NPZ stays on lab machine.\n",
        encoding="utf-8")
    archive = out / "sinr_logs.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in (out / "manifest.json", out / "verification.json", out / "README.txt",
                     native / "sinr.log", native / "sinr_slots.csv", native / "sinr_summary.json",
                     native / "EVAL_COMPLETE.json", native / "provenance.json"):
            z.write(path, path.relative_to(out).as_posix())
    return {**verification, "archive": str(archive), "console_episodes": console_episodes}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-batch", type=Path, required=True)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--worlds", nargs="+", type=int, default=[1001])
    p.add_argument("--console-episodes", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0",
                   help="default matches lab GPU evaluation; CPU has a different stochastic sampling stream")
    p.add_argument("--dry-run", action="store_true", help="metadata/path checks only, no output/model load/rollout")
    args = p.parse_args()
    print(json.dumps(run(args.training_batch, args.batch_id, args.expected_commit, args.worlds,
                         args.console_episodes, args.dry_run, args.device), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
