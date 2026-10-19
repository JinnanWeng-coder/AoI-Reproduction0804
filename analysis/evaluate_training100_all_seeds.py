"""All-seed best100 evaluation; read-only reuse, no new training or seed selection."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import evaluate_training100_selected as selected
from analysis import deployment_contract as dc
from analysis.evaluate_frozen_deployment import csv_read, csv_write
from analysis.deployment_rng import action_seed_words
from analysis.mappo_b1_accounting import T95

CLI_MODULE = 'analysis.evaluate_training100_all_seeds'
ROOT = Path('Training100-all-seeds-eval/P5_N4_gap25')
MODELS = dict(dc.MODELS)
CONDITIONS = dict(selected.CONDITIONS)
SCHEMA = 'training100_all_seeds_eval_v1'
SOURCE_COMMIT = 'ee05154233c9122f5cf6e57e13cf372a60bdf843'


def code_check(expected=None):
    return selected.tc.code_check(expected)


def source_check(source, payload=False):
    return selected.source_check(source, payload, spec=MODELS[source['record']['model_id']])


def make_sources(records, training_batch):
    result = []
    for r in records:
        run = Path(r['run_dir']).resolve()
        dc.require(run.is_relative_to(Path(training_batch).resolve() / 'training'), 'source outside training batch')
        dc.require(Path(r['best']['path']).resolve() == run / selected.tc.BEST, 'wrong accepted best checkpoint')
        result.append({'source_id': f"{r['model_id']}__seed{r['training_seed']:02d}__best100",
                       'checkpoint_kind': 'best100', 'selection_metric': 'best_training100_score',
                       'selection_score': r['best_training100_score'], 'candidate_count': 1,
                       'record': r, 'checkpoint': r['best'],
                       'config_hash': dc.read(run/'COMPLETE.json')['config_hash'],
                       'policy_episode': r['best_episode'], 'policy_version': r['best_policy_version'],
                       'record_file': dc.file_identity(Path(training_batch)/f"records/train_{r['run_id']:02d}.json"),
                       'metadata_files': [dc.file_identity(run/n) for n in
                                          ('config.resolved.json', 'COMPLETE.json', 'training_reward_selection.csv')]})
    return result


def make_units(sources, conditions=None):
    units = []
    for condition, (model, execution, noise, _) in (conditions or CONDITIONS).items():
        for seed in dc.SEEDS:
            source = next(s for s in sources if s['record']['model_id'] == model and s['record']['training_seed'] == seed)
            for world in selected.WORLDS:
                units.append({'unit_id': len(units), 'source_id': source['source_id'], 'model_id': model,
                              'condition': condition+'__best100', 'execution_condition': condition,
                              'checkpoint_kind': 'best100', 'training_seed': seed, 'world': world, 'repeat': 0,
                              'execution': execution, 'noise_std': noise, 'target_scenario': dc.SCENARIO,
                              'action_seed_words': action_seed_words(seed,world,0,MODELS[model]['family'])})
    return units


def unit_key(unit):
    return {k:v for k,v in unit.items() if k not in ('unit_id','source_id')}


def reuse_inventory(old_batch, sources, units):
    old_batch = Path(old_batch).resolve()
    old = dc.read(old_batch/'protocol/manifest.json')
    dc.require(old['schema'] == 'training100_selected_eval_v1' and old['batch_root'] == str(old_batch) and
               old['implementation_commit'] == SOURCE_COMMIT and old['worlds'] == selected.WORLDS and
               old['protocol'] == {'name':'sequential_warm','warmup':5,'scored':100,'slots':100,'repeat':0},
               'wrong source selected evaluation batch/protocol/commit')
    summaries = [p for p in (old_batch/'analysis').glob('a*/verification.json')
                 if dc.read(p).get('technical_status') == 'PASS' and dc.read(p).get('accepted_units') == 300]
    dc.require(summaries and all((old_batch/f'records/unit_{i:03d}.json').is_file() for i in range(300)),
               'finish and accept original 300 units before extending')
    old_sources = dc.read(old_batch/'protocol/sources.json')
    old_units = dc.read(old_batch/'protocol/units.json')
    dc.require(old_units == selected.make_units(old_sources), 'old matrix changed')
    candidates = dc.read(old_batch/'protocol/candidates.json')
    chosen = selected.select(candidates)
    dc.require(len(old_sources) == len(chosen) and all(all(s[k] == v for k,v in c.items())
               for s,c in zip(old_sources, chosen)), 'old selected sources changed')
    mapping = []
    for ou in (u for u in old_units if u['checkpoint_kind'] == 'best100'):
        matches = [u for u in units if unit_key(u) == unit_key(ou)]
        dc.require(len(matches) == 1, 'ambiguous reuse key')
        nu = matches[0]
        osrc = next(s for s in old_sources if s['source_id'] == ou['source_id'])
        nsrc = next(s for s in sources if s['source_id'] == nu['source_id'])
        for key in ('record','checkpoint','config_hash','policy_episode','policy_version','record_file','metadata_files'):
            dc.require(osrc[key] == nsrc[key], f'reuse source mismatch: {key}')
        path = old_batch/f"records/unit_{ou['unit_id']:03d}.json"
        rec = dc.read(path)
        dc.require(rec['technical_status']=='PASS' and rec['unit']==ou and rec['source']==osrc, 'old accepted identity mismatch')
        mapping.append({'unit_id':nu['unit_id'],'old_batch':str(old_batch),'old_unit':ou,'old_source':osrc,
                        'old_manifest':old,'record_file':dc.file_identity(path),
                        'protocol_files':[dc.file_identity(old_batch/'protocol'/n) for n in
                                          ('manifest.json','sources.json','units.json','candidates.json')],
                        'verification_file':dc.file_identity(sorted(summaries)[-1])})
    dc.require(len(mapping)==150 and len({x['unit_id'] for x in mapping})==150, 'exactly 150 reused best100 units required')
    return mapping


def init(training_batch, reuse_batch, batch_id, expected_commit):
    commit = code_check(expected_commit)
    training_batch = Path(training_batch).resolve()
    old = dc.read(training_batch/'protocol/manifest.json')
    study = Path(old['study_root']).resolve()
    dc.require(old['batch_root']==str(training_batch) and training_batch.parent==study/selected.tc.ROOT, 'original training path required')
    dc.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',batch_id), 'unsafe batch ID')
    batch = study/ROOT/batch_id
    dc.require(not batch.exists(), 'unique new batch required')
    records = selected.candidates(training_batch)
    sources = make_sources(records,training_batch)
    configs = [source_check(s).to_dict() for s in sources]
    dc.require(all({k:c[k] for k in dc.ENV_FIELDS}=={k:configs[0][k] for k in dc.ENV_FIELDS} for c in configs), 'environment drift')
    units = make_units(sources)
    mapping = reuse_inventory(reuse_batch,sources,units)
    dc.require(Path(reuse_batch).resolve().parent == study/selected.ROOT, 'reuse outside original study')
    batch.mkdir(parents=True,exist_ok=False)
    for n in ('protocol','units','records','logs','tmp'):
        (batch/n).mkdir()
    m = {'schema':SCHEMA,'batch_root':str(batch),'study_root':str(study),'training_batch':str(training_batch),
         'implementation_commit':commit,'training_commit':old['implementation_commit'],
         'worlds':selected.WORLDS,'world_role':selected.WORLD_ROLE,'new_unseen_confirmation':False,
         'protocol':{'name':'sequential_warm','warmup':5,'scored':100,'slots':100,'repeat':0},
         'policy_world_repeat_count':900,'new_units':750,'reused_units':150,'new_training':0,
         'training_seeds':dc.SEEDS,'checkpoint_kind':'best100','selection_uses_evaluation_results':False}
    for name,data in (('manifest',m),('sources',sources),('candidates',records),('units',units),('reuse',mapping)):
        dc.write(batch/f'protocol/{name}.json',data)
    return m


def load(batch):
    batch = Path(batch).resolve()
    m = dc.read(batch/'protocol/manifest.json')
    code_check(m['implementation_commit'])
    dc.require(m['schema']==SCHEMA and m['batch_root']==str(batch) and batch.parent==Path(m['study_root'])/ROOT and
               m['worlds']==selected.WORLDS and m['training_seeds']==dc.SEEDS and m['checkpoint_kind']=='best100' and
               m['protocol']=={'name':'sequential_warm','warmup':5,'scored':100,'slots':100,'repeat':0} and
               m['world_role']==selected.WORLD_ROLE and m['new_unseen_confirmation'] is False and
               (m['policy_world_repeat_count'],m['new_units'],m['reused_units'])==(900,750,150), 'all-seed contract changed')
    sources = dc.read(batch/'protocol/sources.json')
    records = dc.read(batch/'protocol/candidates.json')
    dc.require(len(records)==24 and {(r['model_id'],r['training_seed']) for r in records}==
               {(model,seed) for model in dc.MODELS for seed in dc.SEEDS},'all 24 seeds required')
    dc.require(sources==make_sources(records,m['training_batch']), 'source manifest changed')
    units = dc.read(batch/'protocol/units.json')
    dc.require(units==make_units(sources), 'all-seed matrix changed')
    reuse = dc.read(batch/'protocol/reuse.json')
    dc.require(len(reuse)==150 and len({x['unit_id'] for x in reuse})==150, 'reuse coverage changed')
    return m,sources,units


def accept(batch,unit,source,manifest,attempt,create=False):
    mapping = next((x for x in dc.read(Path(batch)/'protocol/reuse.json') if x['unit_id']==unit['unit_id']),None)
    if mapping is None:
        return selected.accept(batch,unit,source,manifest,attempt,create)
    dc.require(unit_key(unit)==unit_key(mapping['old_unit']), 'reuse unit identity changed')
    for key in ('record','checkpoint','config_hash','policy_episode','policy_version','record_file','metadata_files'):
        dc.require(source[key]==mapping['old_source'][key], 'reuse policy identity changed')
    for identity in [mapping['record_file'],mapping['verification_file'],*mapping['protocol_files']]:
        dc.require(dc.file_identity(identity['path'])==identity, 'reused evidence changed')
    dc.require(mapping['old_manifest']==dc.read(Path(mapping['old_batch'])/'protocol/manifest.json') and
               mapping['old_manifest']['implementation_commit']==SOURCE_COMMIT, 'reuse manifest changed')
    oldrec = dc.read(mapping['record_file']['path'])
    # Direct pure acceptance deliberately checks the OLD implementation commit;
    # never call old.load(), relabel provenance, or write into the old batch.
    rows,probe = selected.accept(mapping['old_batch'],mapping['old_unit'],mapping['old_source'],
                                 mapping['old_manifest'],oldrec['attempt'])
    rows = [{**r,'unit_id':unit['unit_id'],'source_id':unit['source_id'],'data_role':'reused',
             'reused_from_batch':mapping['old_batch'],'reused_unit_id':mapping['old_unit']['unit_id']} for r in rows]
    rec = {'technical_status':'PASS','unit':unit,'source':source,'attempt':'reuse','data_role':'reused',
           'source_record':mapping['record_file'],'native_completion':oldrec['native_completion'],
           'implementation_commit':manifest['implementation_commit'],
           'source_evaluation_commit':mapping['old_manifest']['implementation_commit'],'row_count':5}
    path = Path(batch)/f"records/unit_{unit['unit_id']:03d}.json"
    if create and not path.exists():
        dc.write(path,rec)
    else:
        dc.require(dc.read(path)==rec,'reuse acceptance changed')
    return rows,probe


def preflight(batch,device='cuda:0'):
    import torch
    m,sources,units = load(batch)
    dc.require(selected.candidates(m['training_batch'])==dc.read(Path(batch)/'protocol/candidates.json'), 'training records changed')
    for source in sources:
        source_check(source,payload=True)
    for entry in dc.read(Path(batch)/'protocol/reuse.json'):
        unit=units[entry['unit_id']]
        source=next(s for s in sources if s['source_id']==unit['source_id'])
        accept(batch,unit,source,m,'reuse',create=True)
    if device=='cuda:0':
        dc.require(torch.cuda.is_available(),'CUDA unavailable')
        a=torch.ones((128,128),device=device)
        dc.require(float((a@a)[0,0])==128,'CUDA probe failed')
        torch.cuda.synchronize()
    report={'technical_status':'PASS','implementation_commit':m['implementation_commit'],'device':device,
            'executable':sys.executable,'loaded_payloads':len(sources),'reused_units':150,'new_units':750,
            'real_rollouts_in_preflight':0,'torch':torch.__version__,'numpy':np.__version__}
    path=Path(batch)/'protocol/preflight.json'
    if path.exists():
        dc.require(dc.read(path)==report,'preflight changed')
    else:
        dc.write(path,report)
    return report


def execute(batch,unit_id,device='cuda:0',attempt='a01',reason=None):
    return selected.execute(batch,unit_id,device,attempt,reason,workflow=sys.modules[__name__])


def paired_tables(seeds):
    groups=defaultdict(dict)
    for row in seeds:
        groups[row['condition']][int(row['training_seed'])]=row
    fields=[k for k in seeds[0] if k.startswith('scene_first_')]
    pairs=[]
    effects=[]
    names=list(groups)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            dc.require(set(groups[a])==set(dc.SEEDS)==set(groups[b]), 'paired seed coverage')
            for metric in fields:
                ds=[]
                for seed in dc.SEEDS:
                    av,bv=groups[a][seed][metric],groups[b][seed][metric]
                    delta=None if av is None or bv is None else av-bv
                    pairs.append({'condition_a':a,'condition_b':b,'training_seed':seed,'metric':metric,'a_minus_b':delta})
                    ds.append(delta)
                valid=all(x is not None for x in ds)
                mean=float(np.mean(ds)) if valid else None
                sd=float(np.std(ds,ddof=1)) if valid else None
                half=T95*sd/math.sqrt(6) if valid else None
                effects.append({'condition_a':a,'condition_b':b,'metric':metric,'n_training_seeds':6,
                                'mean_a_minus_b':mean,'sample_sd':sd,'descriptive_t95_lower':mean-half if valid else None,
                                'descriptive_t95_upper':mean+half if valid else None,'df':5,
                                'positive':sum(x>0 for x in ds) if valid else None,
                                'zero':sum(x==0 for x in ds) if valid else None,
                                'negative':sum(x<0 for x in ds) if valid else None,'claim_support':'NOT_ASSESSED'})
    return pairs,effects


def summarize(batch,attempt='a01'):
    from analysis.summarize_frozen_deployment import aggregate
    m,sources,units=load(batch)
    dc.require(re.fullmatch(r'a\d{2,4}',attempt),'invalid analysis attempt')
    for source in sources:
        source_check(source)
    rows=[]
    probes={}
    native=[]
    for unit in units:
        rec=dc.read(Path(batch)/f"records/unit_{unit['unit_id']:03d}.json")
        source=next(s for s in sources if s['source_id']==unit['source_id'])
        part,probe=accept(batch,unit,source,m,rec['attempt'])
        rows.extend(part)
        if unit['world'] in probes:
            dc.require(probes[unit['world']]==probe,'exogenous probe mismatch')
        probes[unit['world']]=probe
        native.append((unit['unit_id'],Path(rec['native_completion'])))
    n=len(units)
    dc.require(len(rows)==n*5 and sum(r['slot_count'] for r in rows)==n*50000 and
               sum(r['cam_episode_count'] for r in rows)==n*500,'summary denominators')
    dc.require(sum(r['data_role']=='reused' for r in rows)==m['reused_units']*5,'reuse counted incorrectly')
    tables=aggregate(rows)
    dc.require(len(tables['seed_endpoints'])==30 and all(r['world_count']==30 for r in tables['seed_endpoints']), 'six seeds each with 30 worlds')
    dc.require(all(r['n_training_seeds']==6 for r in tables['condition_summary']),'training-seed n must be six')
    pairs,effects=paired_tables(tables['seed_endpoints'])
    tables.update(paired_seed_differences=pairs,paired_effects=effects)
    out=Path(batch)/'analysis'/attempt
    out.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():
        csv_write(out/f'{name}.csv',table)
    verification={'technical_status':'PASS','claim_support':'NOT_ASSESSED','accepted_units':n,
                  'new_units':m['new_units'],'reused_units':m['reused_units'],'platoon_rows':n*5,
                  'scored_agent_slots':n*50000,'cam_deadlines':n*500,'training_seeds':dc.SEEDS,
                  'worlds':selected.WORLDS,'repeat_variability':None,'implementation_commit':m['implementation_commit'],
                  'new_training':0,'exogenous_probe_match':True}
    dc.write(out/'verification.json',verification)
    (out/'REPORT_CN.md').write_text('# 全seed best100冻结评估\n\n'
        '每seed使用自己的best100；不跨seed择优。900=750新增+150只读复用；final不混入。\n'
        '同world车队先按sum/count归约，再分别取AoI最大和CAM最低车队；随后跨world、六training seeds汇总。\n'
        'sample SD与描述性95% t区间以六seed为重复，df=5；配对按seed；不按性能筛选。\n'
        'world1001–1030是已使用的用户固定世界，不是新的未见确认；repeat0不能估计动作repeat变异。\n'
        'legacy另列；缺失不填0。technical PASS仅表示来源/计算通过，claim_support未判定。\n',encoding='utf-8')
    archive=out/'core_evidence.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for root in (out,Path(batch)/'protocol',Path(batch)/'records'):
            for p in root.rglob('*'):
                if p.is_file() and p.suffix in ('.csv','.json','.md'):
                    z.write(p,p.relative_to(batch).as_posix())
        for uid,p in native:
            for name in ('EVAL_COMPLETE.json','provenance.json'):
                z.write(p.with_name(name),f'native_metadata/unit_{uid:03d}/{name}')
        for s in sources:
            for identity in s['metadata_files']:
                p=Path(identity['path'])
                if p.suffix=='.json':
                    z.write(p,f"source_metadata/{s['source_id']}/{p.name}")
    with zipfile.ZipFile(archive) as z:
        dc.require(z.testzip() is None and len(z.namelist())==len(set(z.namelist())),'ZIP integrity/duplicates')
        import csv,io
        table=list(csv.DictReader(io.StringIO(z.read(f'analysis/{attempt}/world_repeat_platoon.csv').decode('utf-8-sig'))))
        dc.require(len(table)==n*5,'ZIP CSV row count')
    return {**verification,'archive':str(archive)}


def run(batch,device='cuda:0',workers=2,ids=None,attempt='a01',reason=None):
    return selected.run(batch,device,workers,ids,attempt,reason,workflow=sys.modules[__name__])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('init')
    for name in ('training-batch','reuse-batch'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-id',required=True)
    p.add_argument('--expected-commit',required=True)
    for name in ('preflight','unit','run','summary'):
        p=sub.add_parser(name)
        p.add_argument('--batch-root',type=Path,required=True)
        if name!='summary':
            p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
        if name!='preflight':
            p.add_argument('--attempt',default='a01')
        if name in ('unit','run'):
            p.add_argument('--reason')
        if name=='unit':
            p.add_argument('--id',type=int,required=True)
        if name=='run':
            p.add_argument('--workers',type=int,choices=(1,2),default=2)
            p.add_argument('--ids')
    a=parser.parse_args(argv)
    if a.command=='init':
        result=init(a.training_batch,a.reuse_batch,a.batch_id,a.expected_commit)
    elif a.command=='preflight':
        result=preflight(a.batch_root,a.device)
    elif a.command=='unit':
        result=execute(a.batch_root,a.id,a.device,a.attempt,a.reason)
    elif a.command=='summary':
        result=summarize(a.batch_root,a.attempt)
    else:
        result=run(a.batch_root,a.device,a.workers,[int(i) for i in a.ids.split(',')] if a.ids else None,a.attempt,a.reason)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
