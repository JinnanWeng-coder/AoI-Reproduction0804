"""One training-reward-selected Shared MADDPG best100, noise0, thirty worlds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

from analysis import deployment_contract as dc, shared_maddpg_contract as training
from analysis import evaluate_training100_selected as selected
from analysis import shared_maddpg_selected_contract as contract
from analysis.evaluate_frozen_deployment import csv_read, csv_write

CLI_MODULE = 'analysis.evaluate_shared_maddpg_selected'
MODELS = {contract.MODEL: contract.SPEC}
ROOT = contract.ROOT


def code_check(expected=None):
    commit = training.code_check(expected)
    dc.require(not dc.git('diff', contract.TRAINING_COMMIT, 'HEAD', '--', 'aoi_v2x_reproduction'),
               'training algorithm/environment changed since shared training commit')
    return commit


def source_check(source, payload=False):
    cfg = selected.source_check(source, payload, spec=contract.SPEC)
    dc.require(cfg.maddpg_actor_sharing is True and cfg.state_dim == 22 and cfg.number_agents == 5,
               'single shared actor with local22 observations required')
    dc.require(source['checkpoint_kind'] == 'best100' and Path(source['checkpoint']['path']).name == selected.tc.BEST and
               source['policy_episode'] == source['record']['best_episode'] and
               source['policy_version'] == source['record']['best_policy_version'] and
               contract.sha256(source['checkpoint']['path']) == source['checkpoint_sha256'], 'selected best checkpoint changed')
    return cfg


def init(training_batch, batch_id, expected_commit):
    commit = code_check(expected_commit)
    parent = Path(training_batch).resolve()
    original = dc.read(parent / 'protocol/manifest.json')
    study = Path(original['study_root']).resolve()
    dc.require(Path(original['batch_root']).resolve() == parent and parent.parent == study / training.ROOT,
               'original live Shared MADDPG training batch required')
    dc.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', batch_id), 'unsafe batch ID')
    batch = study / ROOT / batch_id
    dc.require(not batch.exists(), 'new unique evaluation batch required')
    records = contract.candidates(parent)
    winner = contract.choose(records)
    all_sources = [contract.source_for(record, parent) for record in records]
    for source in all_sources:
        source_check(source)
    sources = [next(s for s in all_sources if s['record'] == winner)]
    units = contract.make_units(sources)
    m = {'schema': contract.SCHEMA, 'batch_root': str(batch), 'study_root': str(study),
         'training_batch': str(parent), 'implementation_commit': commit, 'training_commit': contract.TRAINING_COMMIT,
         'worlds': selected.WORLDS, 'world_role': selected.WORLD_ROLE, 'new_unseen_confirmation': False,
         'protocol': contract.PROTOCOL, 'policy_world_repeat_count': 30, 'platoon_rows': 150,
         'scored_agent_slots': 1500000, 'cam_deadlines': 15000, 'new_training': 0,
         'candidate_training_seeds': dc.SEEDS, 'n_selected_training_seeds_per_condition': 1,
         'selection_ties': 'lowest_training_seed', 'selection_uses_evaluation_results': False,
         'checkpoint_kind': 'best100', 'noise_std': 0.0, 'conditions': [contract.CONDITION]}
    batch.mkdir(parents=True, exist_ok=False)
    for name in ('protocol', 'records', 'units', 'logs', 'tmp'):
        (batch / name).mkdir()
    for name, data in (('manifest', m), ('candidates', records), ('candidate_sources', all_sources),
                       ('sources', sources), ('units', units)):
        dc.write(batch / f'protocol/{name}.json', data)
    csv_write(batch / 'protocol/selected_models.csv', [{
        'model_id': contract.MODEL, 'training_seed': winner['training_seed'], 'candidate_count': 6,
        'selection_score': winner['best_training100_score'], 'window_start': winner['window_start'],
        'window_end': winner['window_end'], 'policy_episode': winner['best_episode'],
        'policy_version': winner['best_policy_version'], 'checkpoint': sources[0]['checkpoint']['path'],
        'checkpoint_sha256': sources[0]['checkpoint_sha256'], 'training_commit': contract.TRAINING_COMMIT,
        'accepted_attempt': Path(winner['run_dir']).parents[1].name, 'inherited_episodes': winner['inherited_episodes']}])
    # Compact evidence is self-contained; no replay, weights or raw training NPZ.
    dc.write(batch / 'protocol/training_manifest.json', original)
    dc.write(batch / 'protocol/training_verification.json', dc.read(parent / 'analysis/a01/verification.json'))
    for name in ('selected_policies.csv', 'training_episode_curves.csv'):
        csv_write(batch / 'protocol' / ('candidate_' + name), csv_read(parent / 'analysis/a01' / name))
    for s in all_sources:
        for item in s['metadata_files']:
            src = Path(item['path'])
            dest = batch / 'protocol/source_metadata' / s['source_id'] / src.name
            if src.suffix == '.json': dc.write(dest, dc.read(src))
            else: csv_write(dest, csv_read(src))
    return m


def load(batch):
    batch = Path(batch).resolve()
    m = dc.read(batch / 'protocol/manifest.json')
    code_check(m['implementation_commit'])
    dc.require(m['schema'] == contract.SCHEMA and m['batch_root'] == str(batch) and
               batch.parent == Path(m['study_root']).resolve() / ROOT and m['worlds'] == selected.WORLDS and
               m['world_role'] == selected.WORLD_ROLE and m['new_unseen_confirmation'] is False and
               m['protocol'] == contract.PROTOCOL and m['policy_world_repeat_count'] == 30 and
               m['conditions'] == [contract.CONDITION] and m['noise_std'] == 0.0 and m['new_training'] == 0 and
               m['selection_uses_evaluation_results'] is False, 'evaluation contract changed')
    records = dc.read(batch / 'protocol/candidates.json')
    winner = contract.choose(records)
    all_sources = dc.read(batch / 'protocol/candidate_sources.json')
    dc.require(len(all_sources) == 6 and [s['record'] for s in all_sources] == records, 'six candidate sources changed')
    for s in all_sources:
        for item in [s['record_file'], *s['metadata_files']]:
            dc.require(dc.file_identity(item['path']) == item, 'candidate metadata changed')
        dc.require(dc.read(s['record_file']['path']) == s['record'], 'candidate accepted record changed')
    sources = dc.read(batch / 'protocol/sources.json')
    dc.require(sources == [next(s for s in all_sources if s['record'] == winner)], 'training-only winner selection changed')
    units = dc.read(batch / 'protocol/units.json')
    dc.require(units == contract.make_units(sources), 'thirty-unit matrix changed')
    return m, sources, units


def preflight(batch, device='cpu'):
    import torch
    m, sources, _ = load(batch)
    dc.require(device == 'cpu', 'this batch uses CPU inference')
    dc.require(contract.candidates(m['training_batch']) == dc.read(Path(batch) / 'protocol/candidates.json'),
               'candidate training evidence changed')
    checks = []
    from aoi_v2x_reproduction.runtime.training_selection import load_training_actors, load_training_policy
    for source in dc.read(Path(batch) / 'protocol/candidate_sources.json'):
        source_check(source, payload=True)
        cfg, data = load_training_policy(source['checkpoint']['path'])
        _, actors = load_training_actors(source['checkpoint']['path'], device='cpu')
        dc.require(len(actors) == 1 and all(not p.requires_grad for p in actors.parameters()) and
                   not hasattr(actors[0], 'optimizer') and data['actor_sharing'] is True and
                   data['actor_network_count'] == 1 and cfg.maddpg_actor_sharing is True,
                   'candidate payload must contain one frozen shared actor without optimizer')
        checks.append({'seed': cfg.seed, 'policy_episode': data['episode'], 'policy_version': data['policy_version'],
                       'actor_network_count': 1, 'checkpoint_sha256': source['checkpoint_sha256']})
    report = {'technical_status': 'PASS', 'implementation_commit': m['implementation_commit'],
              'candidate_runs': 6, 'loaded_candidate_best_payloads': 6, 'selected_payloads': 1,
              'selected_seed': sources[0]['record']['training_seed'], 'selected_episode': sources[0]['policy_episode'],
              'policy_world_repeat_count': 30, 'device': device, 'python': sys.version,
              'executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
              'candidate_checks': checks, 'new_training': 0, 'real_rollouts_in_preflight': 0}
    path = Path(batch) / 'protocol/preflight.json'
    if path.exists(): dc.require(dc.read(path) == report, 'preflight interpreter/device or evidence changed')
    else: dc.write(path, report)
    return report


def native_check(metadata, unit, source, commit):
    selected.native_check(metadata, unit, source, commit, spec=contract.SPEC)
    for key, expected in {'actor_sharing': True, 'actor_network_count': 1,
                          'actor_structure': 'shared', 'model_id': contract.MODEL,
                          'sharing_schema': 'shared_maddpg_tdec_v1', 'critic_constructed': False,
                          'replay_constructed': False,
                          'optimizer_constructed': False, 'wrote_to_source_run': False,
                          'steps_per_episode': 100, 'number_agents': 5, 'n_rb': 3,
                          'explore': False, 'evaluation_device_requested': 'cpu'}.items():
        dc.require(metadata.get(key) == expected, f'native shared {key} mismatch')
    from aoi_v2x_reproduction.config import config_from_dict
    from analysis.evaluate_maddpg_e1 import build_frozen_maddpg_target_config
    cfg = config_from_dict(dc.read(Path(source['record']['run_dir']) / 'config.resolved.json'))
    dc.require(metadata.get('runtime_config') == build_frozen_maddpg_target_config(cfg, dc.SCENARIO, 'cpu').to_dict(),
               'native runtime config mismatch')


def accept(batch, unit, source, manifest, attempt, create=False):
    return selected.accept(batch, unit, source, manifest, attempt, create, native_validator=native_check)


def execute(batch, unit_id, device='cpu', attempt='a01', reason=None):
    dc.require(device == 'cpu', 'this batch uses CPU inference')
    return selected.execute(batch, unit_id, device, attempt, reason, workflow=sys.modules[__name__])


def status(batch, verify=False):
    m, sources, units = load(batch)
    accepted, failed, unrun, native = [], [], [], []
    for u in units:
        record = Path(batch) / f"records/unit_{u['unit_id']:03d}.json"
        if record.exists():
            saved = dc.read(record)
            dc.require(saved['technical_status'] == 'PASS' and saved['unit'] == u and saved['source'] == sources[0],
                       'accepted record identity mismatch')
            if verify: accept(batch, u, sources[0], m, saved['attempt'])
            accepted.append(u['unit_id'])
        else:
            parent = Path(batch) / f"units/unit_{u['unit_id']:03d}"
            if list(parent.rglob('EVAL_COMPLETE.json')): native.append(u['unit_id'])
            elif parent.exists() or list((Path(batch) / 'records').glob(f"process_{u['unit_id']:03d}_*.json")):
                failed.append(u['unit_id'])
            else: unrun.append(u['unit_id'])
    return {'technical_status': 'COMPLETE' if len(accepted) == 30 else 'PARTIAL', 'accepted_units': len(accepted),
            'expected_units': 30, 'accepted_ids': accepted, 'native_completion_pending_acceptance_ids': native,
            'failed_or_partial_ids': failed, 'unrun_ids': unrun, 'arrays_reverified': verify,
            'running': dc.read(Path(batch) / 'RUNNING.json') if (Path(batch) / 'RUNNING.json').exists() else None}


def summarize(batch, attempt='a01'):
    from analysis.summarize_frozen_deployment import aggregate
    m, sources, units = load(batch)
    source_check(sources[0])
    dc.require(re.fullmatch(r'a\d{2,4}', attempt), 'invalid analysis attempt')
    out = Path(batch) / 'analysis' / attempt
    if out.exists():
        marker = out / 'verification.json'
        dc.require(marker.is_file() and dc.read(marker).get('technical_status') == 'PASS',
                   'partial analysis preserved; choose a new analysis attempt')
        return export(batch, attempt)
    rows = []
    for unit in units:
        record = dc.read(Path(batch) / f"records/unit_{unit['unit_id']:03d}.json")
        part, _ = accept(batch, unit, sources[0], m, record['attempt'])
        rows.extend(part)
    dc.require(len(rows) == 150 and sum(r['slot_count'] for r in rows) == 1500000 and
               sum(r['cam_episode_count'] for r in rows) == 15000 and
               len({(r['world'], r['platoon']) for r in rows}) == 150, 'scored coverage mismatch')
    tables = aggregate(rows)
    dc.require(len(tables['world_endpoints']) == 30 and len(tables['seed_endpoints']) == 1 and
               tables['seed_endpoints'][0]['world_count'] == 30 and all(
                   r['n_training_seeds'] == 1 and r['training_seed_sd'] is None and
                   r['descriptive_t95_lower'] is None for r in tables['condition_summary']), 'selected n=1 contract')
    out.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items(): csv_write(out / f'{name}.csv', table)
    result = {'technical_status': 'PASS', 'claim_support': 'NOT_ASSESSED', 'accepted_units': 30,
              'new_units': 30, 'reused_units': 0, 'platoon_rows': 150, 'scored_agent_slots': 1500000,
              'cam_deadlines': 15000, 'warmup_agent_slots_excluded': 75000,
              'selected_training_seeds_per_condition': 1, 'training_seed_uncertainty': None,
              'repeat_variability': None, 'implementation_commit': m['implementation_commit'],
              'selected_seed': sources[0]['record']['training_seed'], 'selected_episode': sources[0]['policy_episode'],
              'selected_checkpoint_sha256': sources[0]['checkpoint_sha256'], 'new_training': 0,
              'worlds': selected.WORLDS, 'world_role': selected.WORLD_ROLE, 'new_unseen_confirmation': False}
    dc.write(out / 'verification.json', result)
    (out / 'REPORT_CN.md').write_text(
        '# Shared MADDPG 训练赢家的冻结评估\n\n'
        '六个候选只按best100训练分数选一个模型，精确并列按最低seed。接受run已经包含继承前缀，不重复拼接。\n'
        '只评估该best权重、noise0、world1001–1030、repeat0，30单元。每world冷启动一次，5 warmup+100 scored×100slots。\n'
        '每world逐车队sum/count后分别取最大AoI与最低CAM；逐world分布和sample SD见world_endpoints、seed_endpoints。\n'
        'legacy先跨world按车队合并再取极值。功率为线性mW；严格AoI>50；reward及分量是轨迹诊断。\n'
        '这是一个训练赢家的30world表现，训练seed不确定性、repeat变异和六seed置信区间为空。\n'
        '这些world已使用，不称新的未见确认。technical PASS与性能方向无关。\n'
        '源权重SHA256、原始路径和best episode/version见protocol；权重及完整NPZ留实验室。\n', encoding='utf-8')
    return export(batch, attempt)


def export(batch, attempt='a01'):
    m, sources, units = load(batch)
    source_check(sources[0])
    out = Path(batch) / 'analysis' / attempt
    dc.require(re.fullmatch(r'a\d{2,4}', attempt) and dc.read(out / 'verification.json')['technical_status'] == 'PASS',
               'summary PASS required')
    dc.require(status(batch, verify=True)['accepted_units'] == 30, 'reverify all units before export')
    archive = out / 'core_evidence.zip'
    if archive.exists():
        verify_archive(archive, attempt)
        return {**dc.read(out / 'verification.json'), 'archive': str(archive),
                'archive_sha256': contract.sha256(archive), 'existing_archive_verified': True}
    runtime = out / 'runtime_report.json'
    if not runtime.exists():
        dc.write(runtime, {'implementation_commit': m['implementation_commit'], 'backend': 'local_python_cpu',
            'processes': [dc.read(p) for p in sorted((Path(batch) / 'records').glob('process_*.json'))],
            'MaxRSS': None, 'raw_evaluation_bytes': sum(p.stat().st_size for p in (Path(batch) / 'units').rglob('*') if p.is_file()),
            'source_checkpoint': sources[0]['checkpoint'], 'source_checkpoint_sha256': sources[0]['checkpoint_sha256']})
    files = {p for root in (out, Path(batch) / 'protocol', Path(batch) / 'records')
             for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.md')}
    for unit in units:
        r = dc.read(Path(batch) / f"records/unit_{unit['unit_id']:03d}.json")
        native = Path(r['native_completion'])
        files.update([native, native.with_name('provenance.json'), native.parent.parent / 'STARTED.json'])
        # Exact per-unit table is small; trajectories remain in metrics.npz.
        files.add(native.parent.parent / 'world_repeat_platoon.csv')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files): z.write(path, path.relative_to(batch).as_posix())
        for path in sorted((Path(batch) / 'logs').glob('*')):
            if path.is_file():
                with path.open('rb') as handle:
                    handle.seek(max(0, path.stat().st_size - 4096))
                    z.writestr('log_excerpts/' + path.name, handle.read())
    verify_archive(archive, attempt)
    return {**dc.read(out / 'verification.json'), 'archive': str(archive), 'archive_sha256': contract.sha256(archive)}


def verify_archive(path, attempt='a01'):
    view = contract.Evidence(path)
    try:
        m = view.json('protocol/manifest.json')
        v = view.json(f'analysis/{attempt}/verification.json')
        dc.require(m['schema'] == contract.SCHEMA and v['technical_status'] == 'PASS' and v['accepted_units'] == 30,
                   'archive verification contract')
        rows = view.csv(f'analysis/{attempt}/world_repeat_platoon.csv')
        dc.require(len(rows) == 150 and sum(int(r['slot_count']) for r in rows) == 1500000 and
                   sum(int(r['cam_episode_count']) for r in rows) == 15000 and
                   len(view.csv(f'analysis/{attempt}/world_endpoints.csv')) == 30, 'archive counts')
        dc.require(all(view.json(f'records/unit_{i:03d}.json')['technical_status'] == 'PASS' for i in range(30)),
                   'archive completion coverage')
        dc.require(not any(n.endswith(('.pt', '.npz')) for n in view.archive.namelist()), 'large artifacts in evidence')
        # The ZIP is adjacent to its immutable analysis directory. Existing ZIPs
        # must still reproduce current compact files, not merely pass CRC/counts.
        live_batch = Path(path).resolve().parents[2]
        if (live_batch / 'protocol/manifest.json').exists():
            for name in view.archive.namelist():
                if name.startswith('log_excerpts/'): continue
                live = live_batch / name
                dc.require(live.is_file() and live.read_bytes() == view.archive.read(name),
                           f'archive compact evidence changed: {name}')
    finally: view.close()


def run(batch, device='cpu', workers=2, ids=None, attempt='a01', reason=None):
    dc.require(device == 'cpu', 'CPU inference required')
    return selected.run(batch, device, workers, ids, attempt, reason, workflow=sys.modules[__name__])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preview')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--training-batch', type=Path)
    g.add_argument('--evidence-zip', type=Path)
    p = sub.add_parser('init')
    p.add_argument('--training-batch', type=Path, required=True)
    p.add_argument('--batch-id', required=True)
    p.add_argument('--expected-commit', required=True)
    for name in ('preflight', 'run', 'unit', 'status', 'summary', 'export'):
        p = sub.add_parser(name)
        p.add_argument('--batch-root', type=Path, required=True)
        if name in ('preflight', 'run', 'unit'): p.add_argument('--device', choices=('cpu',), default='cpu')
        if name in ('run', 'unit', 'summary', 'export'): p.add_argument('--attempt', default='a01')
        if name in ('run', 'unit'): p.add_argument('--reason')
        if name == 'run':
            p.add_argument('--workers', type=int, choices=(1, 2), default=2)
            p.add_argument('--ids', help='explicit failed/unrun IDs, comma separated')
        if name == 'unit': p.add_argument('--id', type=int, required=True)
        if name == 'status': p.add_argument('--verify', action='store_true')
    a = parser.parse_args(argv)
    if a.command == 'preview':
        records = contract.candidates(a.training_batch or a.evidence_zip)
        r = contract.choose(records)
        result = {'candidate_count': 6, 'selected_seed': r['training_seed'], 'best_episode': r['best_episode'],
                  'policy_version': r['best_policy_version'], 'score': r['best_training100_score'],
                  'checkpoint': r['best']['path'], 'units': 30, 'new_training': 0, 'payloads_loaded': False}
    elif a.command == 'init': result = init(a.training_batch, a.batch_id, a.expected_commit)
    elif a.command == 'preflight': result = preflight(a.batch_root, a.device)
    elif a.command == 'unit': result = execute(a.batch_root, a.id, a.device, a.attempt, a.reason)
    elif a.command == 'status': result = status(a.batch_root, a.verify)
    elif a.command == 'summary': result = summarize(a.batch_root, a.attempt)
    elif a.command == 'export': result = export(a.batch_root, a.attempt)
    else:
        ids = [int(x) for x in a.ids.split(',')] if a.ids else None
        result = run(a.batch_root, a.device, a.workers, ids, a.attempt, a.reason)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
