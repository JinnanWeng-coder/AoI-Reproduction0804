"""B1 copy of validated worst-platoon-A-20260925-v1 table accounting.

Source: handoff_20260925/scripts/audit_common.py (through analyze).
No environment, learner, checkpoint, or Git imports. Original G tables untouched.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
from statistics import mean, stdev

VERSION = 'worst-platoon-A-20260925-v1'
SEEDS = tuple(range(8, 14))
ARMS = ('independent', 'shared')
VALUES = ('combined', 'tdec')
KEYS = ('actor_structure', 'value_configuration', 'training_seed', 'world', 'agent')
T95 = 2.570581835636314
SUM_ATOL = 1e-6
MEAN_ATOL = 1e-8
RATE_ATOL = 1e-12
PARTS = ('le50', 'mid50', 'gt100', 'boundary')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(x):
    v = float(x)
    require(math.isfinite(v) and v.is_integer(), f'Not an exact integer: {x!r}')
    return int(v)


def number(x):
    v = float(x)
    require(math.isfinite(v), f'Non-finite number: {x!r}')
    return v


def close(a, b, label, tol=MEAN_ATOL):
    require(abs(number(a) - number(b)) <= tol, f'{label}: {a} != {b} (atol={tol})')


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    require(bool(rows), f'Refusing empty CSV: {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def key(r):
    return (r['actor_structure'], r['value_configuration'].lower(),
            integer(r['training_seed']), integer(r['world']), integer(r['agent']))


def ident(k):
    return dict(zip(KEYS, k))


def unique(rows, key_fn=key):
    result = {}
    for row in rows:
        k = key_fn(row)
        require(k not in result, f'Duplicate key: {k}')
        result[k] = row
    return result


def s100(length):
    require(isinstance(length, int) and length > 0, f'Invalid length {length}')
    return length*(length+1)//2 if length <= 100 else 5050 + 100*(length-100)


def bucket(length):
    return 'le50' if length <= 50 else ('mid50' if length <= 100 else 'gt100')


def summary(values):
    require(len(values) == 6, 'A summary must retain all six training seeds')
    v = [number(x) for x in values]
    m, sd = mean(v), stdev(v)
    half = T95*sd/math.sqrt(6)
    # Exact computed zero, not a data-dependent significance tolerance.
    return dict(n_seeds=6, mean=m, sample_sd=sd, ci95_low=m-half, ci95_high=m+half,
                n_negative=sum(x < 0 for x in v), n_positive=sum(x > 0 for x in v),
                n_zero=sum(x == 0 for x in v))


def exact_aoi_sum(row):
    value = number(row['aoi_sum_ms'])
    close(value, round(value), '1-ms integer AoI sum', SUM_ATOL)
    return round(value)


def compute_flow(service, flow, frequencies):
    """Half-open complete reset cycles; residual includes the final reset slot."""
    n, resets = integer(flow['slot_count']), integer(flow['reset_count'])
    require(n > 0 and 0 <= resets <= n, 'Invalid exposure/reset count')
    require(integer(service['aoi_slot_count']) == n, 'AoI/reset exposure differs')
    total = exact_aoi_sum(service)
    require(n <= total <= 100*n, 'AoI outside [1,100] per slot')
    require(flow['event_source'] in ('reset_event', 'aoi_ms_isclose_1'), 'Unknown event semantics')
    close(flow['reset_rate'], resets/n, 'reset_rate', RATE_ATOL)
    areas = dict.fromkeys(PARTS[:3], 0)
    count = sl = sl2 = long_count = long_sum = max_length = 0
    seen = set()
    for row in frequencies:
        length, freq = integer(row['length_L']), integer(row['frequency'])
        require(length > 0 and freq > 0 and length not in seen, 'Invalid/duplicate interval length')
        seen.add(length)
        count += freq
        sl += length*freq
        sl2 += length*length*freq
        areas[bucket(length)] += s100(length)*freq
        max_length = max(max_length, length)
        if length > 100:
            long_count += freq
            long_sum += length*freq
    require(count == integer(flow['complete_interval_count']) == max(resets-1, 0), 'Interval count mismatch')
    require(sl == integer(flow['sum_L']) and sl2 == integer(flow['sum_L2']), 'Frequency moment mismatch')
    require(long_count == integer(flow['long_gt100_count']), 'Long-interval count mismatch')
    if 'long_gt100_sum_L' in flow:
        require(long_sum == integer(flow['long_gt100_sum_L']), 'Long-interval exposure mismatch')
    if 'no_reset_flow' in flow:
        require(str(flow['no_reset_flow']).lower() == str(resets == 0).lower(), 'No-reset flag mismatch')
    if 'no_complete_interval_flow' in flow:
        require(str(flow['no_complete_interval_flow']).lower() == str(count == 0).lower(), 'No-interval flag mismatch')
    if 'one_reset_flow' in flow:
        require(str(flow['one_reset_flow']).lower() == str(resets == 1).lower(), 'One-reset flag mismatch')
    boundary = total - sum(areas.values())
    boundary_slots = n - sl
    require(boundary_slots <= boundary <= 100*boundary_slots, 'Boundary area violates cap bounds')
    if resets:
        left, right = integer(flow['left_boundary_visible_wait_slots']), integer(flow['right_boundary_visible_wait_slots'])
        require(left >= 0 and right >= 0 and left + sl + right == n-1, 'Boundary identity failed')
        left_area = boundary - s100(right+1)
        require(left <= left_area <= 100*left, 'Post-step left residual violates cap bounds')
    else:
        require(flow['left_boundary_visible_wait_slots'] in ('', None) and flow['right_boundary_visible_wait_slots'] in ('', None), 'No-reset window double counted')
        require(integer(flow['no_reset_window_slots']) == n, 'No-reset exposure mismatch')
    cam_n, cam_k = integer(service['binary_cam_episode_count']), integer(service['binary_cam_success_count'])
    require(cam_n > 0 and 0 <= cam_k <= cam_n, 'Invalid binary CAM counts')
    return dict(slot_count=n, reset_count=resets, aoi_sum_ms=total,
                binary_cam_success_count=cam_k, binary_cam_episode_count=cam_n,
                complete_interval_count=count, sum_L=sl, sum_L2=sl2,
                area_le50=areas['le50'], area_mid50=areas['mid50'], area_gt100=areas['gt100'],
                area_boundary=boundary, boundary_slots=boundary_slots,
                long_gt100_count=long_count, max_complete_interval=max_length if count else '')


def extrema(rows, numerator, denominator, maximum=True):
    ratios = {integer(r['agent']): Fraction(integer(r[numerator]), integer(r[denominator])) for r in rows}
    target = (max if maximum else min)(ratios.values())
    return sorted(i for i, value in ratios.items() if value == target)


def analyze(c1_rows, flow_rows, freq_rows, worlds, world_set):
    c1, flows = unique(c1_rows), unique(flow_rows)
    expected = set(product(ARMS, VALUES, SEEDS, worlds, range(5)))
    require(set(c1) == set(flows) == expected, 'Exact 2x2x6x6x5 coverage/key mismatch')
    freqs = defaultdict(list)
    for r in freq_rows:
        require(key(r) in expected, 'Frequency contains an unknown flow')
        freqs[key(r)].append(r)
    flow_out = []
    for k in sorted(expected):
        require(integer(flows[k]['slot_count']) == 10000 and integer(c1[k]['binary_cam_episode_count']) == 100, 'Scored window mismatch')
        flow_out.append(dict(world_set=world_set, **ident(k), **compute_flow(c1[k], flows[k], freqs[k])))
    agent_out = []
    additive = ('slot_count','reset_count','aoi_sum_ms','binary_cam_success_count','binary_cam_episode_count',
                'complete_interval_count','sum_L','sum_L2','area_le50','area_mid50','area_gt100','area_boundary','boundary_slots','long_gt100_count')
    for arm, val, seed, agent in product(ARMS, VALUES, SEEDS, range(5)):
        selected = [r for r in flow_out if (r['actor_structure'],r['value_configuration'],r['training_seed'],r['agent']) == (arm,val,seed,agent)]
        require(len(selected) == 6, 'Agent/world coverage mismatch')
        r = dict(world_set=world_set, actor_structure=arm,value_configuration=val,training_seed=seed,agent=agent,world_count=6)
        r.update({name: sum(x[name] for x in selected) for name in additive})
        r.update(aoi_ms=r['aoi_sum_ms']/r['slot_count'], reset_rate=r['reset_count']/r['slot_count'],
                 binary_cam=r['binary_cam_success_count']/r['binary_cam_episode_count'])
        r.update({'contrib_'+p: r['area_'+p]/r['slot_count'] for p in PARTS})
        r['contrib_long_complete'] = r['contrib_mid50'] + r['contrib_gt100']
        r['contrib_long_complete_and_boundary'] = r['contrib_long_complete'] + r['contrib_boundary']
        maxima = [x['max_complete_interval'] for x in selected if x['max_complete_interval'] != '']
        r['max_complete_interval'] = max(maxima) if maxima else ''
        agent_out.append(r)
    endpoints, profiles, cams, per_world = [], [], [], []
    for arm,val,seed in product(ARMS,VALUES,SEEDS):
        group = [r for r in agent_out if (r['actor_structure'],r['value_configuration'],r['training_seed']) == (arm,val,seed)]
        ties = extrema(group, 'aoi_sum_ms', 'slot_count')
        cam_ties = extrema(group, 'binary_cam_success_count', 'binary_cam_episode_count', False)
        ranking = sorted(group, key=lambda r: (Fraction(r['aoi_sum_ms'],r['slot_count']), -r['agent']))
        selected = dict(next(r for r in group if r['agent'] == ties[0]))
        selected['aoi_argmax_set'] = '|'.join(map(str,ties))
        selected['tie_count'] = len(ties)
        selected['worst_second_gap_ms'] = ranking[-1]['aoi_ms'] - ranking[-2]['aoi_ms']
        selected['group_mean_aoi_ms'] = sum(r['aoi_sum_ms'] for r in group)/sum(r['slot_count'] for r in group)
        selected['group_reset_rate'] = sum(r['reset_count'] for r in group)/sum(r['slot_count'] for r in group)
        selected['other_four_reset_rate'] = sum(r['reset_count'] for r in group if r['agent'] != ties[0])/sum(r['slot_count'] for r in group if r['agent'] != ties[0])
        tie_rows = [r for r in group if r['agent'] in ties]
        for field in ('reset_rate',)+tuple('contrib_'+p for p in PARTS):
            selected['tie_'+field+'_min'] = min(r[field] for r in tie_rows)
            selected['tie_'+field+'_max'] = max(r[field] for r in tie_rows)
        endpoints.append(selected)
        for rank, r in enumerate(ranking, 1):
            profiles.append(dict(r, rank=rank, is_aoi_argmax=r['agent'] in ties))
        intersection = sorted(set(ties)&set(cam_ties))
        cams.append(dict(world_set=world_set,actor_structure=arm,value_configuration=val,training_seed=seed,
                         aoi_argmax_set='|'.join(map(str,ties)),cam_argmin_set='|'.join(map(str,cam_ties)),
                         cam_argmin_count=len(cam_ties),unique_same_index=(len(ties)==len(cam_ties)==1 and ties==cam_ties),
                         intersection_set='|'.join(map(str,intersection)),any_overlap=bool(intersection),
                         worst_binary_cam=min(r['binary_cam'] for r in group),
                         selected_aoi_agent_cam=selected['binary_cam']))
        for world in worlds:
            wg=[r for r in flow_out if (r['actor_structure'],r['value_configuration'],r['training_seed'],r['world'])==(arm,val,seed,world)]
            ids=extrema(wg,'aoi_sum_ms','slot_count')
            per_world.append(dict(world_set=world_set,actor_structure=arm,value_configuration=val,training_seed=seed,world=world,
                                  aoi_argmax_set='|'.join(map(str,ids)),pooled_argmax_set='|'.join(map(str,ties)),
                                  any_overlap=bool(set(ids)&set(ties)),unique_same_index=len(ids)==len(ties)==1 and ids==ties))
    paired=[]
    fields=('aoi_ms','reset_rate','contrib_le50','contrib_mid50','contrib_gt100','contrib_boundary','contrib_long_complete',
            'contrib_long_complete_and_boundary','group_mean_aoi_ms','group_reset_rate','worst_second_gap_ms')
    for val,seed in product(VALUES,SEEDS):
        a,b=[next(r for r in endpoints if (r['actor_structure'],r['value_configuration'],r['training_seed'])==(arm,val,seed)) for arm in ARMS]
        r=dict(world_set=world_set,value_configuration=val,training_seed=seed,independent_agent=a['agent'],shared_agent=b['agent'])
        for f in fields:
            r['independent_'+f],r['shared_'+f],r['delta_'+f]=a[f],b[f],b[f]-a[f]
        cam=[next(x['worst_binary_cam'] for x in cams if (x['actor_structure'],x['value_configuration'],x['training_seed'])==(arm,val,seed)) for arm in ARMS]
        r.update(independent_worst_cam=cam[0],shared_worst_cam=cam[1],delta_worst_cam_pp=100*(cam[1]-cam[0]))
        close(sum(r['delta_contrib_'+p] for p in PARTS),r['delta_aoi_ms'],'Constructed delta closure')
        paired.append(r)
    levels=[]
    for arm,val in product(ARMS,VALUES):
        rows=[r for r in endpoints if (r['actor_structure'],r['value_configuration'])==(arm,val)]
        for f in fields:
            levels.append(dict(world_set=world_set,actor_structure=arm,value_configuration=val,quantity=f,**summary([r[f] for r in rows])))
    effects=[]
    for val in VALUES:
        rows=[r for r in paired if r['value_configuration']==val]
        for f in tuple('delta_'+f for f in fields)+('delta_worst_cam_pp',):
            effects.append(dict(world_set=world_set,value_configuration=val,quantity=f,**summary([r[f] for r in rows])))
    maxima=[]
    for arm,val in product(ARMS,VALUES):
        for scope,rows in [('all_flows',[r for r in flow_out if (r['actor_structure'],r['value_configuration'])==(arm,val)]),
                           ('own_worst_aoi_agents',[r for r in endpoints if (r['actor_structure'],r['value_configuration'])==(arm,val)])]:
            values=[r['max_complete_interval'] for r in rows if r['max_complete_interval']!='']
            maxima.append(dict(world_set=world_set,actor_structure=arm,value_configuration=val,scope=scope,
                               max_complete_interval=max(values) if values else '',long_gt100_count=sum(r['long_gt100_count'] for r in rows)))
    return dict(flows=flow_out,agents=agent_out,endpoints=endpoints,profiles=profiles,cams=cams,per_world=per_world,
                paired=paired,levels=levels,effects=effects,maxima=maxima)
