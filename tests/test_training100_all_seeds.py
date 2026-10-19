"""Synthetic full-seed manifests and reuse; no real policies/environments."""
import copy
from pathlib import Path

import pytest

from analysis import evaluate_training100_all_seeds as ae
from analysis import evaluate_training100_selected as se
from tests.test_training100_selected_eval import fake_training_batch
from tests.test_frozen_deployment import dump, synthetic_arrays, META


@pytest.fixture
def batches(tmp_path, monkeypatch):
    training, records = fake_training_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(ae, 'SOURCE_COMMIT', 'eval_commit')
    old = Path(se.init(training,'selected_old','eval_commit')['batch_root'])
    m,sources,units = se.load(old)
    for u in units:
        s=next(s for s in sources if s['source_id']==u['source_id'])
        native=old/f"units/unit_{u['unit_id']:03d}/a01/native/EVAL_COMPLETE.json"
        dump(native, {'status':'complete'})
        dump(native.with_name('provenance.json'), {'synthetic':True})
        dump(old/f"records/unit_{u['unit_id']:03d}.json", {'technical_status':'PASS','unit':u,'source':s,'attempt':'a01','native_completion':str(native)})
    dump(old/'analysis/a01/verification.json',{'technical_status':'PASS','accepted_units':300})
    return training,old,records


def test_all_seeds_matrix_and_exact_reuse(batches):
    training,old,records=batches
    out=Path(ae.init(training,old,'all_test','eval_commit')['batch_root'])
    m,sources,units=ae.load(out)
    assert len(sources)==24 and len(units)==900
    assert len({(u['condition'],u['training_seed'],u['world'],u['repeat']) for u in units})==900
    assert len({s['checkpoint']['path'] for s in sources})==24
    assert all(u['checkpoint_kind']=='best100' and u['repeat']==0 for u in units)
    reuse=ae.dc.read(out/'protocol/reuse.json')
    assert len(reuse)==150 and all(x['old_unit']['training_seed']==13 for x in reuse)
    assert m['new_units']==750 and m['reused_units']==150
    assert 'a02' in sources[2]['record']['run_dir']
    with pytest.raises(ValueError,match='unique'):
        ae.init(training,old,'all_test','eval_commit')
    bad=copy.deepcopy(units)
    bad[0]['world']=999
    p=out/'protocol/units.json'
    p.unlink(); dump(p,bad)
    with pytest.raises(ValueError,match='matrix'):
        ae.load(out)


def test_reuse_read_only_old_commit_and_source_mutation_rejected(batches,monkeypatch):
    training,old,_=batches
    out=Path(ae.init(training,old,'reuse_test','eval_commit')['batch_root'])
    m,sources,units=ae.load(out)
    entry=ae.dc.read(out/'protocol/reuse.json')[0]
    unit=units[entry['unit_id']]
    source=next(s for s in sources if s['source_id']==unit['source_id'])
    snapshot={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    calls=[]
    def old_accept(b,u,s,m,a,create=False):
        calls.append((str(b),u,s,m,a,create))
        return [{'unit_id':u['unit_id'],'source_id':s['source_id'],'data_role':'new',
                 'implementation_commit':m['implementation_commit']}], {'world':u['world']}
    monkeypatch.setattr(se,'accept',old_accept)
    rows,_=ae.accept(out,unit,source,m,'reuse',create=True)
    assert calls[0][-1] is False and calls[0][0]==str(old)
    assert rows[0]['data_role']=='reused' and rows[0]['unit_id']==unit['unit_id']
    assert rows[0]['implementation_commit']=='eval_commit'
    assert snapshot=={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    assert ae.accept(out,unit,source,m,'reuse')[0]==rows
    p=Path(entry['record_file']['path']); p.write_text('{}')
    with pytest.raises(ValueError,match='reused evidence changed'):
        ae.accept(out,unit,source,m,'reuse')


def test_full_summary_six_seeds_and_compact_zip(batches,monkeypatch):
    training,old,_=batches
    out=Path(ae.init(training,old,'summary_test','eval_commit')['batch_root'])
    m,sources,units=ae.load(out)
    reused={x['unit_id'] for x in ae.dc.read(out/'protocol/reuse.json')}
    # Only one small set of synthetic arrays is reduced, then counters reused.
    template=se.reduce_arrays(synthetic_arrays(episodes=100),{**META,'eval_seeds':[1001]},{'world':1001})
    def fake_accept(b,u,s,m,a,create=False):
        return [{**r,**u,'data_role':'reused' if u['unit_id'] in reused else 'new'} for r in template], {'world':u['world'],'initial':'fixed','final':'fixed'}
    monkeypatch.setattr(ae,'accept',fake_accept)
    native=old/'units/unit_000/a01/native/EVAL_COMPLETE.json'
    for u in units:
        dump(out/f"records/unit_{u['unit_id']:03d}.json", {'attempt':'reuse' if u['unit_id'] in reused else 'a01','native_completion':str(native)})
    v=ae.summarize(out)
    assert (v['accepted_units'],v['new_units'],v['reused_units'])==(900,750,150)
    assert v['scored_agent_slots']==45000000 and v['cam_deadlines']==450000
    rows=ae.csv_read(out/'analysis/a01/condition_summary.csv')
    assert all(r['n_training_seeds']=='6' and r['df']=='5' for r in rows)
    assert len(ae.csv_read(out/'analysis/a01/world_repeat_platoon.csv'))==4500
    assert len(ae.csv_read(out/'analysis/a01/seed_endpoints.csv'))==30
    with pytest.raises(FileExistsError):
        ae.summarize(out)


def test_missing_original_completion_blocks_extension(batches):
    training,old,_=batches
    (old/'records/unit_299.json').unlink()
    with pytest.raises(ValueError,match='finish and accept'):
        ae.init(training,old,'blocked','eval_commit')
    assert not (training.parents[2]/ae.ROOT/'blocked').exists()
