"""Six fresh Shared Modified MADDPG-TDec runs; explicit new research contract."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from analysis import deployment_contract as dc, training100_contract as tc
from analysis import evaluate_training100_selected as selected
from analysis.evaluate_frozen_deployment import csv_read,csv_write
from analysis.mappo_e5_contract import config_differences
from aoi_v2x_reproduction.config import config_from_dict

CLI_MODULE='analysis.shared_maddpg_contract'
ROOT=Path('Shared-MADDPG-TDec/P5_N4_gap25')
SCHEMA='shared_maddpg_training_v1'
BASELINE='ee05154233c9122f5cf6e57e13cf372a60bdf843'
MODEL='shared_maddpg_tdec'
# Reviewed extension only; never disable the old training100/D1 guards.
CORE_ALLOWLIST={
    'aoi_v2x_reproduction/config.py',
    'aoi_v2x_reproduction/algorithms/modified_maddpg/agent.py',
    'aoi_v2x_reproduction/algorithms/modified_maddpg/learner.py',
    'aoi_v2x_reproduction/algorithms/modified_maddpg/networks.py',
    'aoi_v2x_reproduction/runtime/checkpointing.py',
    'aoi_v2x_reproduction/runtime/training_selection.py',
    'aoi_v2x_reproduction/runtime/runner.py',
    'aoi_v2x_reproduction/runtime/shared_resume.py',
}


def code_check(expected=None):
    commit=dc.git('rev-parse','HEAD')
    dc.require(dc.git('branch','--show-current') in ('research/mappo','') and
               not dc.git('status','--porcelain','--untracked-files=all'),'pinned clean research checkout required')
    dc.require(expected is None or expected==commit,'implementation commit changed')
    changes=set(dc.git('diff','--name-only',BASELINE,'HEAD','--','aoi_v2x_reproduction').splitlines())
    dc.require(changes<=CORE_ALLOWLIST,f'outside reviewed shared-MADDPG extension: {changes-CORE_ALLOWLIST}')
    return commit


def cell(i):
    dc.require(0<=i<6,'Shared MADDPG ID must be 0..5')
    return {'run_id':i,'model_id':MODEL,'training_seed':8+i,'episodes':2500,'slots':100,'environment_steps':250000}


def check_source(s):
    for identity in s['parent_files']:
        dc.require(dc.file_identity(identity['path'])==identity,'parent source changed')
    dc.require(dc.read(s['parent_files'][0]['path'])==s['parent_record'] and
               dc.read(s['parent_files'][1]['path'])==s['config'],'parent record/config changed')
    r=s['parent_record']
    dc.require(r['model_id']=='algorithm2' and r['training_seed']==s['training_seed'] and
               r['technical_status']=='PASS' and r['environment_steps']==250000,'parent treatment mismatch')


def derive(s,output_root):
    cfg=config_from_dict(s['config'])
    before=cfg.to_dict()
    dc.require(cfg.algorithm=='modified_maddpg_tdec' and not cfg.maddpg_actor_sharing and
               cfg.episodes==2500 and cfg.steps_per_episode==100 and cfg.number_agents==5 and cfg.n_rb==3 and
               cfg.seed==s['training_seed'] and cfg.scenario.id==dc.SCENARIO and cfg.exploration_noise==.3 and
               cfg.global_actor_weight==1 and cfg.global_update_mode=='synchronous_joint' and
               cfg.checkpoint_mode=='policy_only' and cfg.eval_protocol=='sequential_warm' and cfg.eval_warmup_episodes==5,
               'independent algorithm2 parent configuration mismatch')
    cfg.maddpg_actor_sharing=True
    cfg.run_name=f'train100_{MODEL}_seed{cfg.seed:02d}'
    cfg.output_root=str(Path(output_root).resolve())
    delta=config_differences(before,cfg.to_dict())
    dc.require(set(delta)=={'maddpg_actor_sharing','run_name','output_root'},'shared derivation changes unintended factors')
    return cfg,delta


def init(training_batch,batch_id,expected_commit):
    commit=code_check(expected_commit)
    training_batch=Path(training_batch).resolve()
    original=dc.read(training_batch/'protocol/manifest.json')
    study=Path(original['study_root']).resolve()
    dc.require(original['batch_root']==str(training_batch) and training_batch.parent==study/tc.ROOT,'original training batch required')
    dc.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',batch_id),'unsafe batch ID')
    batch=study/ROOT/batch_id
    dc.require(not batch.exists(),'unique new batch required')
    records=selected.candidates(training_batch)
    sources=[]
    for r in records:
        if r['model_id']!='algorithm2': continue
        run=Path(r['run_dir'])
        sources.append({'model_id':MODEL,'training_seed':r['training_seed'],'parent_record':r,
                        'config':dc.read(run/'config.resolved.json'),
                        'parent_files':[dc.file_identity(training_batch/f"records/train_{r['run_id']:02d}.json"),
                                        dc.file_identity(run/'config.resolved.json'),dc.file_identity(run/'COMPLETE.json')]})
    dc.require(len(sources)==6,'six parent seeds required')
    derived=[]
    for s in sources:
        check_source(s)
        cfg,delta=derive(s,batch/'training/a01/runs')
        derived.append({'seed':s['training_seed'],'config':cfg.to_dict(),'differences':delta})
    batch.mkdir(parents=True,exist_ok=False)
    for n in ('protocol','records','training','evaluation','analysis','logs','tmp','cache'):
        (batch/n).mkdir()
    m={'schema':SCHEMA,'batch_root':str(batch),'study_root':str(study),'implementation_commit':commit,
       'baseline_commit':BASELINE,'training_parent':str(training_batch),'rule':tc.RULE,
       'runs':[cell(i) for i in range(6)],'new_training_runs':6,'training_env_steps':1500000,
       'source_mode':'accepted_independent_configs','resume_supported':True,'resume_interval':50,'new_evaluation_units':0,
       'final_evaluation_units':0,'concurrent_training_processes':1,'actor_objective':'(global_loss+sum_leader_task1_task2_losses)/P'}
    dc.write(batch/'protocol/manifest.json',m)
    dc.write(batch/'protocol/sources.json',sources)
    dc.write(batch/'protocol/derived_configs.json',derived)
    return m


def load(batch):
    batch=Path(batch).resolve()
    m=dc.read(batch/'protocol/manifest.json')
    code_check(m['implementation_commit'])
    sources=dc.read(batch/'protocol/sources.json')
    dc.require(m['schema']==SCHEMA and m['batch_root']==str(batch) and batch.parent==Path(m['study_root'])/ROOT and
               m['rule']==tc.RULE and m['runs']==[cell(i) for i in range(6)] and
               m.get('resume_supported') is True and m.get('resume_interval')==50 and
               m.get('new_evaluation_units')==0 and m.get('concurrent_training_processes')==1 and
               len(sources)==6 and {(s['model_id'],s['training_seed']) for s in sources}=={(MODEL,s) for s in dc.SEEDS},
               'shared training contract changed')
    derived=[]
    for s in sources:
        check_source(s)
        cfg,delta=derive(s,batch/'training/a01/runs')
        derived.append({'seed':s['training_seed'],'config':cfg.to_dict(),'differences':delta})
    dc.require(dc.read(batch/'protocol/derived_configs.json')==derived,'derived config snapshot changed')
    return m,sources


def attempt_run(batch,i,attempt):
    dc.require(re.fullmatch(r'a\d{2,4}',attempt),'unsafe attempt')
    return Path(batch)/'training'/attempt/'runs'/f"train100_{MODEL}_seed{cell(i)['training_seed']:02d}"


def verify_run(batch,i,run):
    record=tc.verify_run(batch,i,run,contract=sys.modules[__name__])
    complete=dc.read(Path(run)/'COMPLETE.json')
    cfg=config_from_dict(dc.read(Path(run)/'config.resolved.json'))
    updates=(250000-cfg.batch_size+1)//cfg.policy_delay
    dc.require(complete.get('actor_sharing') is True and complete.get('model_id')==MODEL and
               complete['actor_network_count']==complete['target_actor_network_count']==1 and
               complete['actor_optimizer_step_count']==complete['target_actor_update_count']==updates and
               complete['learn_step_count']==250000-cfg.batch_size+1,'shared actor/target update accounting')
    learning=dc.read(Path(run)/'learning_diagnostics.json')
    dc.require(len(learning)==2500 and sum(r['actor_update_count'] for r in learning)==updates,
               'learning diagnostics actor update count')
    recovery=complete.get('episode_resume',{})
    dc.require(recovery.get('schema')=='shared_training100_episode_resume_v1' and recovery.get('interval')==50,
               'missing periodic recovery contract')
    resumed_from=recovery.get('resume_source')
    resumed_episode=int(recovery.get('resume_episode',0))
    if resumed_from:
        started=dc.read(Path(batch)/f'records/started_{i:02d}_{Path(run).parents[1].name}.json')
        dc.require(started['resume_source']==resumed_from and
                   started['resume_source_identity']==dc.file_identity(resumed_from) and 0<resumed_episode<2500,
                   'resume lineage changed')
    else:
        dc.require(resumed_episode==0,'fresh run has nonzero inherited episodes')
    return {**record,'actor_structure':'shared','actor_network_count':1,'actor_optimizer_step_count':updates,
            'target_actor_update_count':updates,'learn_step_count':complete['learn_step_count'],
            'from_scratch':not bool(resumed_from),'resume_source':resumed_from,'inherited_episodes':resumed_episode,
            'attempt_environment_steps':(2500-resumed_episode)*100}


def train_run(batch,i,attempt='a01',reason=None,*,resume=None):
    from aoi_v2x_reproduction.runtime.runner import train
    m,sources=load(batch)
    item=cell(i)
    pre=dc.read(Path(batch)/'protocol/preflight.json')
    dc.require(pre['technical_status']=='PASS' and pre['python_executable']==sys.executable and
               pre['implementation_commit']==m['implementation_commit'],'same-environment preflight required')
    path=Path(batch)/f'records/train_{i:02d}.json'
    if path.exists():
        dc.require(resume is None,'accepted seed cannot be retrained/resumed for reselection')
        saved=dc.read(path)
        dc.require(verify_run(batch,i,saved['run_dir'])==saved,'accepted training changed')
        return {'skipped_complete':True,**saved}
    run=attempt_run(batch,i,attempt)
    dc.require(attempt=='a01' or reason,'retry requires reason')
    possible=list((Path(batch)/'training').glob(f'a*/runs/{run.name}/COMPLETE.json'))
    dc.require(len(possible)<=1,'duplicate complete training runs')
    if possible:
        dc.require(resume is None,'completed native run must be verified, not resumed')
        result=verify_run(batch,i,possible[0].parent)
    else:
        dc.require(not run.exists(),'partial training preserved; resume or retry must use a new attempt')
        cfg,delta=derive(next(s for s in sources if s['training_seed']==item['training_seed']),run.parent)
        if resume is not None:
            checkpoint=Path(resume).resolve()
            dc.require(reason and checkpoint.is_relative_to((Path(batch)/'training').resolve()) and
                       checkpoint.parent.name=='checkpoints' and checkpoint.parent.parent.name==run.name,
                       'resume requires reason and this seed/batch checkpoint')
        dc.write(Path(batch)/f'records/started_{i:02d}_{attempt}.json',
                 {**item,'attempt':attempt,'reason':reason,'derivation':delta,'implementation_commit':m['implementation_commit'],
                  'resume_source':str(Path(resume).resolve()) if resume else None,
                  'resume_source_identity':dc.file_identity(resume) if resume else None})
        train(cfg,training_reward_best=True,shared_resume_interval=50,resume=str(resume) if resume else None)
        result=verify_run(batch,i,run)
    code_check(m['implementation_commit'])
    dc.write(path,result)
    return result


def preflight(batch):
    import torch
    from analysis.run_training100_local import cuda_probe
    m,sources=load(batch)
    dc.require(sys.version_info[:2]==(3,10) and torch.__version__.split('+')[0]=='2.11.0' and torch.version.cuda=='12.6',
               'reuse the established Python3.10 torch2.11.0 cu126 environment')
    versions={}
    for line in (dc.REPO/'requirements.cuda.lock.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        name,expected=line.strip().split('==')
        actual=importlib.metadata.version(name)
        dc.require(actual==expected,f'dependency mismatch: {name}')
        versions[name]=actual
    for source in sources:
        cfg,_=derive(source,Path(batch)/'training/a01/runs')
        dc.require(cfg.device=='cuda:0','CUDA config required')
    result={'technical_status':'PASS','implementation_commit':m['implementation_commit'],
            'python_executable':sys.executable,'python':sys.version,'dependencies':versions,'gpu':cuda_probe(),
            'source_configs':6,'new_training_runs':6,'real_rollouts_in_preflight':0}
    path=Path(batch)/'protocol/preflight.json'
    if path.exists(): dc.require(dc.read(path)==result,'preflight environment changed')
    else: dc.write(path,result)
    return result


def summary(batch,attempt='a01'):
    m,_=load(batch)
    dc.require(re.fullmatch(r'a\d{2,4}',attempt),'unsafe analysis attempt')
    records=[]; curves=[]
    for i in range(6):
        r=dc.read(Path(batch)/f'records/train_{i:02d}.json')
        dc.require(verify_run(batch,i,r['run_dir'])==r,'accepted training changed')
        records.append(r)
        curves.extend({**cell(i),**row} for row in csv_read(Path(r['run_dir'])/'training_reward_selection.csv'))
    dc.require(len(curves)==15000,'training curve coverage')
    out=Path(batch)/'analysis'/attempt
    out.mkdir(parents=True,exist_ok=False)
    csv_write(out/'selected_policies.csv',records)
    csv_write(out/'training_episode_curves.csv',curves)
    result={'technical_status':'PASS','claim_support':'NOT_ASSESSED','accepted_runs':6,'best_policies':6,'final_policies':6,
                  'episode_rows':15000,'training_env_steps':1500000,'new_evaluations':0,
                  'implementation_commit':m['implementation_commit']}
    dc.write(out/'verification.json',result)
    return result


def accepted_candidates(batch):
    m,_=load(batch)
    completed=[p for p in (Path(batch)/'analysis').glob('a*/verification.json') if dc.read(p).get('technical_status')=='PASS']
    dc.require(completed,'six-run training summary required')
    records=[]
    for i in range(6):
        r=dc.read(Path(batch)/f'records/train_{i:02d}.json')
        dc.require(verify_run(batch,i,r['run_dir'])==r,'accepted training changed')
        records.append(r)
    return records


def export(batch,attempt='a01'):
    m,_=load(batch)
    out=Path(batch)/'analysis'/attempt
    dc.require(dc.read(out/'verification.json')['technical_status']=='PASS','training summary required')
    archive=out/'core_evidence.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for root in (Path(batch)/'protocol',Path(batch)/'records',out):
            for p in root.rglob('*'):
                if p.is_file() and p.suffix in ('.csv','.json','.md'):
                    z.write(p,p.relative_to(batch).as_posix())
        for i in range(6):
            r=dc.read(Path(batch)/f'records/train_{i:02d}.json')
            for name in ('COMPLETE.json','config.resolved.json','provenance.json','learning_diagnostics.json','training_reward_selection.csv'):
                p=Path(r['run_dir'])/name
                z.write(p,p.relative_to(batch).as_posix())
        for p in (Path(batch)/'logs').glob('*.out'):
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-8192))
                z.writestr('log_excerpts/'+p.name,f.read())
    verify_archive(archive,attempt)
    return {'archive':str(archive),'technical_status':'PASS'}


def verify_archive(path,attempt='a01'):
    with zipfile.ZipFile(path) as z:
        dc.require(z.testzip() is None and len(set(z.namelist()))==len(z.namelist()),'ZIP integrity')
        for name,n in (('selected_policies',6),('training_episode_curves',15000)):
            rows=list(csv.DictReader(io.StringIO(z.read(f'analysis/{attempt}/{name}.csv').decode('utf-8-sig'))))
            dc.require(len(rows)==n,'ZIP training CSV coverage')
        dc.require(len([n for n in z.namelist() if re.fullmatch(r'records/train_\d{2}.json',n)])==6,'ZIP accepted records')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('init','preflight','run','train','summary','export'))
    p.add_argument('--training-batch',type=Path)
    p.add_argument('--batch-root',type=Path)
    p.add_argument('--batch-id')
    p.add_argument('--expected-commit')
    p.add_argument('--id',type=int)
    p.add_argument('--ids')
    p.add_argument('--attempt',default='a01')
    p.add_argument('--reason')
    p.add_argument('--resume',type=Path,help='train only: retained episode checkpoint, new attempt required')
    a=p.parse_args(argv)
    dc.require(a.resume is None or a.command=='train','resume is only supported for an explicit train cell')
    if a.command=='init':
        dc.require(a.training_batch and a.batch_id and a.expected_commit,'init requires training-batch, batch-id, expected-commit')
        result=init(a.training_batch,a.batch_id,a.expected_commit)
    else:
        dc.require(a.batch_root,'batch-root required')
        if a.command=='preflight': result=preflight(a.batch_root)
        elif a.command=='train': result=train_run(a.batch_root,a.id,a.attempt,a.reason,resume=a.resume)
        elif a.command=='summary': result=summary(a.batch_root,a.attempt)
        elif a.command=='export': result=export(a.batch_root,a.attempt)
        else:
            from analysis.run_training100_local import run_queue
            result=run_queue(a.batch_root,a.ids,a.attempt,a.reason,workflow=sys.modules[__name__])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
