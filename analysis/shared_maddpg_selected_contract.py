"""Read-only training evidence and locked single-winner deployment contract."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from analysis import deployment_contract as dc, training100_contract as tc
from analysis import evaluate_training100_selected as selected
from aoi_v2x_reproduction.config import config_from_dict

MODEL = 'shared_maddpg_tdec'
TRAINING_COMMIT = 'c459115c58d7a465f3a4f2358a6385c649e42019'
ROOT = Path('Shared-MADDPG-selected-eval/P5_N4_gap25')
SCHEMA = 'shared_maddpg_selected_best100_eval_v1'
SPEC = {'family': 'maddpg', 'structure': 'shared', 'variant': 'tdec'}
CONDITION = 'shared_maddpg_tdec_noise0__best100'
PROTOCOL = {'name': 'sequential_warm', 'warmup': 5, 'scored': 100, 'slots': 100, 'repeat': 0}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Evidence:
    """Read a live batch or compact ZIP without extraction or path relocation."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.archive:
            names = self.archive.namelist()
            dc.require(len(names) == len(set(names)) and self.archive.testzip() is None, 'invalid evidence ZIP')

    def text(self, name):
        dc.require(not Path(name).is_absolute() and '..' not in name.replace('\\', '/').split('/'), 'unsafe member')
        data = self.archive.read(name) if self.archive else (self.path / name).read_bytes()
        return data.decode('utf-8-sig')

    def json(self, name): return json.loads(self.text(name))
    def csv(self, name): return list(csv.DictReader(io.StringIO(self.text(name))))
    def close(self):
        if self.archive: self.archive.close()


def relative_run(record, manifest):
    root = manifest['batch_root'].replace('\\', '/').rstrip('/')
    run = record['run_dir'].replace('\\', '/')
    dc.require(run.startswith(root + '/training/'), 'accepted run outside original training batch')
    result = run[len(root) + 1:]
    dc.require('..' not in result.split('/') and len(result.split('/')) == 4 and
               result.split('/')[1].startswith('a') and result.split('/')[2] == 'runs', 'invalid accepted attempt path')
    return result


def choose(records):
    dc.require(len(records) == 6 and {(r['model_id'], r['training_seed']) for r in records} ==
               {(MODEL, seed) for seed in dc.SEEDS}, 'six Shared MADDPG candidate seeds required')
    dc.require(all(np.isfinite(float(r['best_training100_score'])) for r in records), 'nonfinite selection score')
    return min(records, key=lambda r: (-float(r['best_training100_score']), int(r['training_seed'])))


def candidates(path):
    view = Evidence(path)
    try:
        m = view.json('protocol/manifest.json')
        v = view.json('analysis/a01/verification.json')
        dc.require(m['schema'] == 'shared_maddpg_training_v1' and m['implementation_commit'] == TRAINING_COMMIT and
                   m['rule'] == tc.RULE and m['runs'] == [
                       {'run_id': i, 'model_id': MODEL, 'training_seed': i+8, 'episodes': 2500,
                        'slots': 100, 'environment_steps': 250000} for i in range(6)], 'training manifest mismatch')
        dc.require(v['technical_status'] == 'PASS' and v['accepted_runs'] == 6 and v['episode_rows'] == 15000 and
                   v['training_env_steps'] == 1500000 and v['implementation_commit'] == TRAINING_COMMIT,
                   'accepted six-run summary required')
        summary = view.csv('analysis/a01/selected_policies.csv')
        curves = view.csv('analysis/a01/training_episode_curves.csv')
        dc.require(len(summary) == 6 and {int(r['run_id']) for r in summary} == set(range(6)) and
                   len(curves) == 15000 and {int(r['run_id']) for r in curves} == set(range(6)), 'training table coverage')
        records = []
        configs = []
        for i in range(6):
            r = view.json(f'records/train_{i:02d}.json')
            dc.require(all(r[k] == value for k, value in m['runs'][i].items()) and
                       r['technical_status'] == 'PASS' and r['implementation_commit'] == TRAINING_COMMIT and
                       r['actor_structure'] == 'shared' and r['actor_network_count'] == 1, 'accepted candidate identity')
            run = relative_run(r, m)
            dc.require(r['best']['path'].replace('\\', '/') == r['run_dir'].replace('\\', '/') + '/' + tc.BEST,
                       'best must come from accepted run, including inherited prefix')
            cfg = config_from_dict(view.json(run + '/config.resolved.json'))
            complete = view.json(run + '/COMPLETE.json')
            configs.append(cfg.to_dict())
            dc.require(cfg.algorithm == 'modified_maddpg_tdec' and cfg.maddpg_actor_sharing is True and
                       cfg.seed == r['training_seed'] and cfg.episodes == 2500 and cfg.steps_per_episode == 100 and
                       cfg.state_dim == 22 and cfg.number_agents == 5 and cfg.n_rb == 3 and cfg.scenario.id == dc.SCENARIO and
                       cfg.global_actor_weight == 1 and cfg.eval_protocol == 'sequential_warm' and
                       cfg.eval_warmup_episodes == 5, 'shared candidate config mismatch')
            dc.require(complete['status'] == 'complete' and complete['reproduction_git_commit'] == TRAINING_COMMIT and
                       complete['config_hash'] == cfg.canonical_hash() and complete['episodes'] == 2500 and
                       complete['actor_sharing'] is True and complete['model_id'] == MODEL and
                       complete['actor_network_count'] == complete['target_actor_network_count'] == 1,
                       'shared completion/config mismatch')
            selection = complete['training_reward_selection']
            dc.require(selection['rule'] == tc.RULE and selection['final_episode'] == 2500, 'best100 rule changed')
            own = view.csv(run + '/training_reward_selection.csv')
            merged = sorted((row for row in curves if int(row['run_id']) == i), key=lambda row: int(row['episode']))
            dc.require(len(own) == len(merged) == 2500 and
                       [int(row['episode']) for row in own] == list(range(1, 2501)) and
                       [int(row['episode']) for row in merged] == list(range(1, 2501)), 'full inherited episode ledger required once')
            rewards = np.asarray([float(row['episode_reward']) for row in own], dtype=np.float64)
            dc.require(np.isfinite(rewards).all() and all(
                row['model_id'] == MODEL and int(row['training_seed']) == r['training_seed'] and
                float(row['episode_reward']) == reward for row, reward in zip(merged, rewards)), 'training curve mismatch')
            episode, score, rolling = tc.best_index(rewards)
            dc.require(all(row['rolling100'] == '' for row in own[:99]) and
                       all(float(row['rolling100']) == value for row, value in zip(own[99:], rolling)), 'rolling100 recomputation mismatch')
            dc.require(r['best_episode'] == episode == selection['best']['selected_episode'] and
                       r['best_training100_score'] == score == selection['best']['score'] and
                       r['last100_training_score'] == float(np.mean(rewards[-100:], dtype=np.float64)) and
                       r['window_start'] == selection['best']['window_start'] == episode-99 and
                       r['window_end'] == selection['best']['window_end'] == episode and
                       r['best_policy_version'] == selection['best']['policy_version'] ==
                       max(0, episode*100-cfg.batch_size+1)//cfg.policy_delay and
                       r['save_timing'] == selection['best']['save_timing'] == 'after_episode_online_actor_updates',
                       'best episode/window/version mismatch')
            inherited = int(r['inherited_episodes'])
            recovery = complete['episode_resume']
            dc.require(recovery['resume_episode'] == inherited and recovery['resume_source'] == r['resume_source'] and
                       r['from_scratch'] == (inherited == 0) and r['attempt_environment_steps'] == (2500-inherited)*100,
                       'inherited prefix lineage mismatch')
            old = next(row for row in summary if int(row['run_id']) == i)
            for key in ('model_id', 'run_dir', 'implementation_commit'):
                dc.require(old[key] == r[key], 'selection summary identity mismatch')
            for key in ('training_seed', 'best_episode', 'best_policy_version', 'best_training100_score', 'last100_training_score'):
                dc.require(float(old[key]) == r[key], 'selection summary score mismatch')
            records.append(r)
        dc.require(len({r['run_dir'] for r in records}) == 6, 'duplicate accepted runs')
        env = {k: configs[0][k] for k in dc.ENV_FIELDS}
        dc.require(all({k: cfg[k] for k in dc.ENV_FIELDS} == env for cfg in configs), 'candidate physical environment drift')
        choose(records)
        return records
    finally:
        view.close()


def source_for(record, training_batch):
    run = Path(record['run_dir']).resolve()
    return {'source_id': f"{MODEL}__seed{record['training_seed']:02d}__best100", 'checkpoint_kind': 'best100',
            'selection_metric': 'best_training100_score', 'selection_score': record['best_training100_score'],
            'candidate_count': 6, 'record': record, 'checkpoint': record['best'],
            'checkpoint_sha256': sha256(record['best']['path']),
            'config_hash': dc.read(run / 'COMPLETE.json')['config_hash'],
            'policy_episode': record['best_episode'], 'policy_version': record['best_policy_version'],
            'record_file': dc.file_identity(Path(training_batch) / f"records/train_{record['run_id']:02d}.json"),
            'metadata_files': [dc.file_identity(run / name) for name in
                               ('config.resolved.json', 'COMPLETE.json', 'training_reward_selection.csv')]}


def make_units(sources):
    dc.require(len(sources) == 1, 'one selected source required')
    s = sources[0]
    seed = s['record']['training_seed']
    return [{'unit_id': i, 'source_id': s['source_id'], 'model_id': MODEL, 'condition': CONDITION,
             'execution_condition': 'shared_maddpg_tdec_noise0', 'checkpoint_kind': 'best100',
             'training_seed': seed, 'world': world, 'repeat': 0, 'execution': 'deterministic', 'noise_std': 0.0,
             'target_scenario': dc.SCENARIO, 'action_seed_words': selected.action_seed_words(seed, world, 0, 'maddpg')}
            for i, world in enumerate(selected.WORLDS)]
