"""Adapter crossover on saved natural observations, never fed back into games."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from undercover_bench.runner import ModalPolicy
from experiments.character_channel_eval import contextualize, CONSTITUTION, backup


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for arm in ['matched-base','matched-character']:
        folder=root.parent/(root.name+'-'+arm)
        ledger=next((folder/'episodes').glob('*.jsonl'))
        rows=[json.loads(l) for l in ledger.read_text().splitlines()]
        candidates=[r for r in rows if r['kind']=='decision' and r['phase']=='report' and r['observation']['private_inbox']]
        if not candidates:raise ValueError('No natural observable invitation for '+arm)
        row=candidates[0]
        cases.append({'origin':arm,'source_ledger':str(ledger),'source_prefix_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),
            'source_sequence':row['sequence'],'observation':contextualize(row['observation']),
            'legal':row['legal'],'seed':row['model_metadata']['sampling_seed'],
            'original_action':row['action']})
    protocol={'purpose':'natural_observation_adapter_crossover','cases':cases,'conditions':[False,True],
        'rlcr_updates':0,'feedback_to_games':False,'independent_games':0,
        'limitation':'Post-hoc diagnostic selected after a reporting discrepancy. Two related observations from one matched game pair; local adapter sensitivity only, not a held-out population estimate.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    with (out/'responses.jsonl').open('x') as f:
        for c in cases:
            for enabled in (False,True):
                policy=ModalPolicy('glm_flash',CONSTITUTION,'M2_candidate_engineering' if enabled else 'M0c',character_candidate=enabled)
                # Replay the original per-decision seed directly, not a new episode RNG.
                policy.next_decision_seed=lambda observation,seed=c['seed']:seed
                response=policy.decide(c['observation'],c['legal'])
                row={'origin':c['origin'],'character':enabled,'response':response,'original_action':c['original_action']}
                f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno());records.append(row)
                print(json.dumps({'origin':c['origin'],'character':enabled,'action':response['action'],'evidence_ids':response.get('evidence_ids'),'rationale':response.get('rationale')}),flush=True)
                backup(out)
    (out/'result.json').write_text(json.dumps([{'origin':r['origin'],'character':r['character'],'action':r['response']['action'],'evidence_ids':r['response'].get('evidence_ids'),'rationale':r['response'].get('rationale')} for r in records],indent=2));backup(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.root,args.out)
