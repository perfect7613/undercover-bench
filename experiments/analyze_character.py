"""Describe frozen character checks with case-clustered, not token-level, units."""
import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def rows_for(folder):
    rows=[]
    for path in sorted(Path(folder).glob('auditor_final_v8_*/trajectories.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def case_means(rows,metric=None):
    grouped=defaultdict(list)
    for row in rows:
        if row.get('error'):
            raise ValueError('infrastructure-incomplete checks cannot support a behavioral comparison')
        key=row.get('logs',{}).get('case_id') or row['example_id']
        grouped[key].append(float(row['reward'] if metric is None else row['metrics'][metric]))
    if not grouped or any(len(values)!=2 for values in grouped.values()):
        raise ValueError('frozen character comparison requires two recorded samples per case')
    return {key:statistics.fmean(values) for key,values in grouped.items()}


def paired_case_bootstrap(left,right,samples=5000,seed=123456):
    if set(left)!=set(right):
        raise ValueError('paired comparison case sets differ')
    differences=[left[key]-right[key] for key in sorted(left)]
    rng=random.Random(seed)
    boot=sorted(statistics.fmean(rng.choices(differences,k=len(differences))) for _ in range(samples))
    return {'point':statistics.fmean(differences),'ci95':[boot[int(.025*samples)],boot[int(.975*samples)]],
            'clusters':len(differences),'bootstrap_samples':samples,
            'limitation':'conditional on these few templated cases; no training-seed uncertainty; sampling RNG not paired'}


def analyze(run):
    run=Path(run);conditions={};means={};artifacts={}
    for condition in ['base','constitution','dpo','sft']:
        folder=run/'frozen_final_v8'/condition
        rows=rows_for(folder)
        if len(rows)!=16:raise ValueError('all four frozen conditions must have 16 recorded episodes')
        means[condition]=case_means(rows)
        conditions[condition]={'episodes':len(rows),'cases':len(means[condition]),
            'overall_correct':sum(r['reward']==1 for r in rows),
            'work_complete':sum(r['metrics']['work_complete'] for r in rows),
            'supported_reports_by_id_coverage':sum(r['metrics']['supported_report'] for r in rows),
            'false_reports_by_id_coverage':sum(r['metrics']['false_report'] for r in rows),
            'missed_reports_by_id_coverage':sum(r['metrics']['missed_report'] for r in rows),
            'episodes_with_invalid_action':sum(r['metrics']['invalid_action'] for r in rows)}
        for path in folder.glob('auditor_final_v8_*/trajectories.jsonl'):
            artifacts[str(path.relative_to(run))]=hashlib.sha256(path.read_bytes()).hexdigest()
    result={'schema':1,'conditions':conditions,
        'comparisons':{'sft_minus_base':paired_case_bootstrap(means['sft'],means['base']),
                       'sft_minus_full_constitution':paired_case_bootstrap(means['sft'],means['constitution']),
                       'sft_minus_dpo':paired_case_bootstrap(means['sft'],means['dpo'])},
        'report_prose_independently_verified':False,'character_improvement_established':False,
        'interpretation':'Frozen results require separate inspection of reporting, work and invalid actions. '
            'Citation coverage alone does not verify report prose. These small case-clustered checks do not '
            'automatically promote a character intervention or establish broad generalization.',
        'artifacts':artifacts}
    (run/'character_comparison.json').write_text(json.dumps(result,indent=2))
    checkpoint=json.loads((run/'sft/checkpoint.json').read_text())
    evidence={'checkpoint':checkpoint['sampler_path'],'held_out':True,'independent_of_training':True,
        'reviewer':'deterministic controller scores and assistant analysis; no independent human review',
        'improvement_established':False,'controls_saturated':True,
        'reason':'Evidence-ID-based reporting controls are saturated; no reporting gain; structured-action regressions observed.',
        'artifacts':artifacts,'results':conditions,'broad_character_generalization_established':False}
    (run/'character_learning_evidence.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('runs/character/paper-v7'))
    analyze(parser.parse_args().run_dir)
