"""Synthetic portable actors/metadata and fake rollout arrays; no formal run."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from analysis import evaluate_shared_maddpg_selected as ev
from analysis import shared_maddpg_selected_contract as c
from analysis import evaluate_maddpg_e1 as md
from analysis.evaluate_frozen_deployment import csv_write
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.algorithms.modified_maddpg.networks import ActorNetwork
from aoi_v2x_reproduction.runtime import checkpointing as cp, training_selection as ts
from tests.test_frozen_deployment import synthetic_arrays, META, FakeEnvironment


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def training(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, 'code_check', lambda expected=None: 'eval_commit')
    monkeypatch.setattr(cp, '_git_stamp', lambda: {'reproduction_git_commit': c.TRAINING_COMMIT,
                       'reproduction_git_dirty': False, 'reproduction_git_branch': None})
    study = tmp_path / 'study'
    batch = study / ev.training.ROOT / 'synthetic_shared'
    runs = [ev.training.cell(i) for i in range(6)]
    put(batch / 'protocol/manifest.json', {'schema': 'shared_maddpg_training_v1', 'batch_root': str(batch),
        'study_root': str(study), 'implementation_commit': c.TRAINING_COMMIT, 'rule': ts.RULE, 'runs': runs})
    records, merged = [], []
    episodes = [336, 232, 229, 124, 957, 1781]
    centers = [-.27, -.20, -.17, -.26, -.25, -.18]
    for i in range(6):
        seed = i+8
        inherited = 1850 if i < 4 else 0
        attempt = 'a02' if inherited else 'a01'
        run = ev.training.attempt_run(batch, i, attempt)
        run.mkdir(parents=True)
        cfg = resolve_config(algorithm='modified_maddpg_tdec', scenario='p05_n04_g25', seed=seed,
            episodes=2500, steps_per_episode=100, device='cpu', actor_hidden=[8, 4],
            maddpg_actor_sharing=True, output_root=str(run.parent), run_name=run.name)
        cfg.profile = 'paper_faithful'  # Real historical profile remains valid for frozen use.
        rewards = np.full(2500, -1., np.float64)
        rewards[episodes[i]-100:episodes[i]] = centers[i]
        best_ep, best_score, rolling = ev.selected.tc.best_index(rewards)
        version = max(0, best_ep*100-cfg.batch_size+1)//cfg.policy_delay
        final_version = max(0, 250000-cfg.batch_size+1)//cfg.policy_delay
        best = {'rule': ts.RULE, 'selected_episode': best_ep, 'window_start': best_ep-99, 'window_end': best_ep,
                'score': best_score, 'policy_version': version, 'save_timing': 'after_episode_online_actor_updates',
                'window_policy_version_min': max(0, (best_ep-100)*100-cfg.batch_size+1)//cfg.policy_delay,
                'window_policy_version_max': version}
        resume_source = str(ev.training.attempt_run(batch, i, 'a01') / 'checkpoints/episode_001850.pt') if inherited else None
        selection = {'rule': ts.RULE, 'best': best, 'final_episode': 2500, 'final_policy_version': final_version}
        put(run / 'config.resolved.json', cfg.to_dict())
        put(run / 'COMPLETE.json', {'status': 'complete', 'algorithm': cfg.algorithm, 'episodes': 2500,
            'config_hash': cfg.canonical_hash(), 'reproduction_git_commit': c.TRAINING_COMMIT,
            'training_reward_selection': selection, 'actor_sharing': True, 'actor_network_count': 1,
            'target_actor_network_count': 1, 'model_id': c.MODEL,
            'episode_resume': {'resume_episode': inherited, 'resume_source': resume_source}})
        actor = ActorNetwork(cfg.state_dim, cfg.actor_hidden, cfg.action_dim, cfg.actor_lr, 'cpu', create_optimizer=False)
        for filename, ep, role, ver in ((ts.BEST, best_ep, 'best_training100', version),
                                       ('policy_final.pt', 2500, 'final_training100', final_version)):
            payload = cp.build_policy_payload(cfg, [SimpleNamespace(actor=actor)], ep)
            payload.update(training_completed=True, checkpoint_role=role, selected_episode=ep, policy_version=ver,
                           training_reward_selection=best if filename == ts.BEST else {'rule': ts.RULE, 'policy_version': ver})
            torch.save(payload, run / filename)
        own = [{'episode': e, 'episode_reward': float(rewards[e-1]),
                'rolling100': rolling[e-100] if e >= 100 else None} for e in range(1, 2501)]
        csv_write(run / 'training_reward_selection.csv', own)
        merged.extend({**runs[i], **row} for row in own)
        record = {**runs[i], 'technical_status': 'PASS', 'implementation_commit': c.TRAINING_COMMIT,
            'run_dir': str(run), 'actor_structure': 'shared', 'actor_network_count': 1, 'from_scratch': not bool(inherited),
            'inherited_episodes': inherited, 'resume_source': resume_source, 'attempt_environment_steps': (2500-inherited)*100,
            'best': ev.dc.file_identity(run / ts.BEST), 'final': ev.dc.file_identity(run / 'policy_final.pt'),
            'best_episode': best_ep, 'window_start': best_ep-99, 'window_end': best_ep,
            'best_training100_score': best_score, 'last100_training_score': float(np.mean(rewards[-100:])),
            'best_policy_version': version, 'final_policy_version': final_version,
            'save_timing': 'after_episode_online_actor_updates'}
        records.append(record)
        put(batch / f'records/train_{i:02d}.json', record)
    (batch / 'analysis/a01').mkdir(parents=True)
    csv_write(batch / 'analysis/a01/selected_policies.csv', records)
    csv_write(batch / 'analysis/a01/training_episode_curves.csv', merged)
    put(batch / 'analysis/a01/verification.json', {'technical_status': 'PASS', 'accepted_runs': 6,
        'episode_rows': 15000, 'training_env_steps': 1500000, 'implementation_commit': c.TRAINING_COMMIT})
    return batch, records


def new_batch(training):
    parent, _ = training
    return Path(ev.init(parent, 'eval_test', 'eval_commit')['batch_root'])


def test_six_candidates_inherited_best_and_only_thirty_units(training):
    parent, records = training
    before = {p: p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    assert c.candidates(parent) == records
    assert c.choose(records)['training_seed'] == 10
    batch = new_batch(training)
    m, sources, units = ev.load(batch)
    s = sources[0]
    assert (s['record']['training_seed'], s['policy_episode'], s['policy_version']) == (10, 229, 11418)
    assert s['record']['inherited_episodes'] == 1850 and '/a02/' in s['checkpoint']['path'].replace('\\', '/')
    assert len(units) == 30 and len({(u['world'], u['repeat']) for u in units}) == 30
    assert all(u['checkpoint_kind'] == 'best100' and u['noise_std'] == 0 for u in units)
    assert m['new_unseen_confirmation'] is False and m['new_training'] == 0
    assert ev.preflight(batch)['loaded_candidate_best_payloads'] == 6
    assert before == {p: p.read_bytes() for p in parent.rglob('*') if p.is_file()}


def test_selection_tie_and_missing_seed():
    records = [{'model_id': c.MODEL, 'training_seed': s, 'best_training100_score': -.1} for s in range(8, 14)]
    assert c.choose(list(reversed(records)))['training_seed'] == 8
    with pytest.raises(ValueError, match='six'): c.choose(records[:-1])
    records[0]['best_training100_score'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'): c.choose(records)


def test_candidate_zip_preview_no_weights_or_output(training, tmp_path, capsys):
    parent, _ = training
    archive = tmp_path / 'synthetic_metadata.zip'
    with zipfile.ZipFile(archive, 'x') as z:
        for p in parent.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.csv'): z.write(p, p.relative_to(parent).as_posix())
    ev.main(['preview', '--evidence-zip', str(archive)])
    result = json.loads(capsys.readouterr().out)
    assert result['selected_seed'] == 10 and result['best_episode'] == 229
    assert result['units'] == 30 and result['payloads_loaded'] is False


@pytest.mark.parametrize('mutation', ['duplicate_prefix', 'wrong_episode', 'wrong_sharing'])
def test_corrupt_candidate_metadata_blocks_before_output(training, mutation):
    parent, records = training
    run = Path(records[2]['run_dir'])
    if mutation == 'duplicate_prefix':
        rows = ev.csv_read(run / 'training_reward_selection.csv')
        (run / 'training_reward_selection.csv').unlink()
        csv_write(run / 'training_reward_selection.csv', rows[:1850] + rows)
    else:
        data = ev.dc.read(run / 'COMPLETE.json')
        if mutation == 'wrong_episode': data['training_reward_selection']['best']['selected_episode'] = 1850
        else: data['actor_sharing'] = False
        put(run / 'COMPLETE.json', data)
    with pytest.raises(ValueError): ev.init(parent, 'bad_test', 'eval_commit')
    assert not (parent.parents[2] / ev.ROOT / 'bad_test').exists()


def test_five_leaders_share_actor_but_use_their_own_observation(training, monkeypatch):
    batch = new_batch(training)
    _, sources, _ = ev.load(batch)
    path = Path(sources[0]['checkpoint']['path'])
    cfg, _ = ts.load_training_policy(path)
    cfg.device = 'cpu'
    monkeypatch.setattr(md, '_make_system', lambda *a: pytest.fail('no critic/replay construction'))
    _, agents, learner, _ = md._training_policy_system(path, cfg)
    assert len(agents) == 5 and len({id(a.actor) for a in agents}) == 1 and learner is None
    actor = agents[0].actor
    assert not hasattr(actor, 'optimizer') and all(not p.requires_grad for p in actor.parameters())
    observed = []
    def hook(module, args): observed.append(args[0].detach().clone())
    handle = actor.register_forward_pre_hook(hook)
    before = md._parameter_snapshot(agents, learner)
    obs = np.arange(110, dtype=np.float32).reshape(5, 22) / 110
    actions = [a.choose_action(obs[i], explore=False, noise_std=0) for i, a in enumerate(agents)]
    handle.remove()
    assert len(observed) == 5 and np.asarray(actions).shape == (5, 3)
    for i, value in enumerate(observed): np.testing.assert_array_equal(value.numpy()[0], obs[i])
    assert md._parameters_unchanged(agents, learner, before)


def test_reused_native_rollout_frozen_and_best_episode_not_resume_or_final(training, tmp_path, monkeypatch):
    batch = new_batch(training)
    m, sources, units = ev.load(batch)
    path = Path(sources[0]['checkpoint']['path'])
    source = sources[0]
    import aoi_v2x_reproduction.envs.platoon as env_module
    monkeypatch.setattr(env_module, 'PaperEnviron', FakeEnvironment)
    monkeypatch.setattr(md, '_make_system', lambda *a: pytest.fail('no training system'))
    monkeypatch.setattr(md, '_git_metadata', lambda: {'reproduction_git_commit': 'eval_commit', 'reproduction_git_dirty': False})
    override = {'run_dir': path.parent, 'resolved_run_dir': path.parent, 'checkpoint': path,
        'canonical_run_name': path.parent.name, 'historical_run_name': path.parent.name,
        'algorithm_label': c.MODEL, 'is_symlink': False, 'complete': ev.dc.read(path.parent / 'COMPLETE.json'), 'manifest': {}}
    before = path.read_bytes()
    result = md.evaluate_maddpg_e1('modified_maddpg_tdec', 0, 10, tmp_path, tmp_path / 'fake_rollout', 'cpu',
        [910001], 1, 5, 229, target_scenario=ev.dc.SCENARIO, source_override=override,
        deployment=True, world_role=ev.selected.WORLD_ROLE, training_policy=True)
    assert result['checkpoint_episode'] == 229 and result['policy_version'] == 11418
    assert result['actor_sharing'] and result['actor_network_count'] == 1 and result['parameter_snapshot_unchanged']
    assert result['optimizer_steps'] == 0 and result['optimizer_constructed'] is False and result['critic_constructed'] is False
    assert path.read_bytes() == before
    # Native protocol checks use fixed formal dimensions; the actual test ran
    # one fake scored episode on a non-formal world, never the research matrix.
    formal = {**result, 'eval_seeds': [1001], 'environment_seeds': [1001], 'eval_episodes': 100,
              'action_seed_words': [units[0]['action_seed_words']],
              'exogenous_probes': [{'world': 1001, 'initial': {}, 'final': {}}]}
    ev.native_check(formal, units[0], source, m['implementation_commit'])
    with pytest.raises(ValueError, match='checkpoint_episode'):
        ev.native_check({**formal, 'checkpoint_episode': 1850}, units[0], source, m['implementation_commit'])


def native_unit(batch, unit, source, m):
    dest = batch / f"units/unit_{unit['unit_id']:03d}/a01"
    native = dest / 'native'
    native.mkdir(parents=True)
    put(dest / 'STARTED.json', {'unit': unit, 'source': source, 'implementation_commit': m['implementation_commit']})
    cfg = ev.dc.read(Path(source['record']['run_dir']) / 'config.resolved.json')
    # Preserve source config exactly, except runtime device/formal flag as the evaluator does.
    cfg['device'], cfg['is_formal_result'] = 'cpu', False
    meta = {**META, 'status': 'complete', 'training_seed': unit['training_seed'], 'eval_seeds': [unit['world']],
        'environment_seeds': [unit['world']], 'action_repeat': 0, 'action_seed_words': [unit['action_seed_words']],
        'eval_protocol': 'sequential_warm', 'eval_warmup_episodes': 5, 'eval_episodes': 100,
        'target_scenario': ev.dc.SCENARIO, 'source_scenario': ev.dc.SCENARIO, 'source_config_hash': source['config_hash'],
        'training_source_commit': c.TRAINING_COMMIT, 'reproduction_git_commit': m['implementation_commit'],
        'reproduction_git_dirty': False, 'world_role': ev.selected.WORLD_ROLE, 'checkpoint_role': 'best_training100',
        'policy_version': 11418, 'training_budget_episodes': 2500, 'checkpoint_episode': 229,
        'parameter_snapshot_unchanged': True, 'optimizer_steps': 0, 'normalization_updated': False, 'noise_std': 0.,
        'checkpoint_name': ts.BEST, 'checkpoint': source['checkpoint']['path'], 'actor_sharing': True,
        'actor_network_count': 1, 'actor_structure': 'shared', 'model_id': c.MODEL, 'sharing_schema': 'shared_maddpg_tdec_v1',
        'critic_constructed': False, 'replay_constructed': False, 'optimizer_constructed': False,
        'wrote_to_source_run': False, 'steps_per_episode': 100, 'number_agents': 5, 'n_rb': 3, 'explore': False,
        'evaluation_device_requested': 'cpu', 'runtime_config': cfg,
        'raw_metric_axes': {'aoi_ms': ['eval_seed', 'scored_episode', 'slot', 'agent']},
        'exogenous_probes': [{'world': unit['world'], 'initial': {}, 'final': {}}]}
    put(native / 'EVAL_COMPLETE.json', meta)
    put(native / 'provenance.json', {'reproduction_git_commit': m['implementation_commit']})
    np.savez_compressed(native / 'metrics.npz', **synthetic_arrays(episodes=100))
    return dest


def test_accept_existing_native_skip_and_partial_retry(training):
    batch = new_batch(training)
    m, sources, units = ev.load(batch)
    source, unit = sources[0], units[0]
    native_unit(batch, unit, source, m)
    ev.preflight(batch)
    assert ev.execute(batch, 0)['accepted_existing_native']
    assert ev.execute(batch, 0)['skipped_verified']
    state = ev.status(batch, verify=True)
    assert state['accepted_units'] == 1 and state['unrun_ids'] == list(range(1, 30))
    (batch / 'units/unit_001/a01').mkdir(parents=True)
    with pytest.raises(ValueError, match='partial attempt preserved'): ev.execute(batch, 1)
    with pytest.raises(ValueError, match='retry requires'): ev.execute(batch, 1, attempt='a02')


def test_thirty_unit_summary_counts_scene_first_and_archive(training, monkeypatch):
    batch = new_batch(training)
    m, sources, units = ev.load(batch)
    # One real synthetic NPZ is reused as a fixture to keep disk use small.
    for u in units:
        native_unit(batch, u, sources[0], m)
        ev.accept(batch, u, sources[0], m, 'a01', create=True)
    result = ev.summarize(batch)
    assert (result['accepted_units'], result['platoon_rows'], result['scored_agent_slots'], result['cam_deadlines']) == (30, 150, 1500000, 15000)
    assert len(result['archive_sha256']) == 64 and result['selected_seed'] == 10
    world = ev.csv_read(batch / 'analysis/a01/world_endpoints.csv')
    per_agent = ev.csv_read(batch / 'analysis/a01/world_repeat_platoon.csv')[:5]
    assert float(world[0]['worst_aoi_ms']) == max(float(r['aoi_sum_ms'])/int(r['slot_count']) for r in per_agent)
    assert float(world[0]['worst_cam_success']) == min(float(r['cam_deadline_success_count'])/int(r['cam_episode_count']) for r in per_agent)
    cond = ev.csv_read(batch / 'analysis/a01/condition_summary.csv')
    assert all(r['n_training_seeds'] == '1' and r['training_seed_sd'] == '' and r['descriptive_t95_lower'] == '' for r in cond)
    assert ev.export(batch)['archive_sha256'] == result['archive_sha256']
    assert ev.summarize(batch)['archive_sha256'] == result['archive_sha256']
    (batch / 'analysis/a02').mkdir()
    with pytest.raises(ValueError, match='partial analysis preserved'):
        ev.summarize(batch, 'a02')
    with zipfile.ZipFile(result['archive']) as z:
        assert not any(n.endswith(('.pt', '.npz')) for n in z.namelist())
        assert 'protocol/source_metadata/shared_maddpg_tdec__seed10__best100/config.resolved.json' in z.namelist()
        assert len([n for n in z.namelist() if n.endswith('EVAL_COMPLETE.json')]) == 30


def test_final_cannot_replace_selected_best_and_weight_mutation_rejected(training):
    batch = new_batch(training)
    _, sources, _ = ev.load(batch)
    source = copy.deepcopy(sources[0])
    source['checkpoint'] = source['record']['final']
    source['checkpoint_sha256'] = c.sha256(source['checkpoint']['path'])
    with pytest.raises(ValueError, match='best checkpoint'):
        ev.source_check(source)
    path = Path(sources[0]['checkpoint']['path'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='source file changed'):
        ev.source_check(sources[0])


def test_old_matrix_stays_three_hundred():
    records = [{'model_id': model, 'training_seed': seed, 'best_training100_score': float(seed),
                'last100_training_score': -float(seed)} for model in ev.dc.MODELS for seed in ev.dc.SEEDS]
    units = ev.selected.make_units(ev.selected.select(records))
    assert len(units) == 300 and len({u['condition'] for u in units}) == 10
