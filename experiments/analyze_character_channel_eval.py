"""Evidence-preserving descriptive comparison for the frozen channel-rule pilot."""
import argparse
import hashlib
import json
from pathlib import Path

from undercover_bench.runner import decision_prompt
from experiments.character_channel_eval import contextualize, CONSTITUTION, backup


def analyze(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    comparisons={}
    for job in manifest['jobs']:
        folder=root.parent/(root.name+'-'+job['id'])
        result_path=folder/'analysis.json'
        if not result_path.exists():
            comparisons[job['id']]={'status':'pending'}
            continue
        value=json.loads(result_path.read_text())
        if 'error' in value:
            comparisons[job['id']]=value
            continue
        ledger=Path(value['ledger']);rows=[json.loads(l) for l in ledger.read_text().splitlines()]
        ds=[r for r in rows if r['kind']=='decision']
        # Reconstruct exactly what this frozen policy supplied. Clearly label it
        # as reconstructed, rather than replacing the logged controller observation.
        prompts=folder/'reconstructed_inputs.jsonl'
        with prompts.open('w') as stream:
            for row in ds:
                observation=contextualize(row['observation'],job['backgrounds'])
                prompt=decision_prompt(observation,row['legal'])
                if row['player']==job['auditor']:
                    prompt=CONSTITUTION+'\n\n'+prompt
                stream.write(json.dumps({'kind':'reconstructed_model_input','sequence':row['sequence'],
                    'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                    'source_ledger_sha256':value['sha256']})+'\n')
        reasons=[{'sequence':r['sequence'],'phase':r['phase'],'action':r['action'],'rationale':r.get('rationale')}
                 for r in ds if r['player']==job['auditor'] and r['phase'] in ('nominate','vote','report','channel_offer','challenge')]
        comparisons[job['id']]={'status':'complete' if value['end']['completed'] else 'truncated',
            'winner':value['end']['winner'],'invalid_actions':value['end']['invalid_actions'],
            'decisions':value['decisions'],'channel_behavior':value['channel_behavior'],
            'channel_rule_violations_observed':sum(s['offers']+s['accepts']+s['whispers'] for s in value['channel_behavior'].values()),
            'visible_messages_to_auditor':value['auditor_visible_message_ids'],
            'reports_delivered':len(value['human_report_receipts']),
            'auditor_reasons':reasons,'ledger':str(ledger),'ledger_sha256':value['sha256']}
        backup(folder)
    result={'schema':1,'rlcr_updates':0,'conditions':comparisons,
        'character_improvement_established':False,
        'interpretation':'Descriptive only. No conclusion about character improvement follows from one matched game pair, unobserved violations, victory alone, verbosity or suspicion language. Inspect rationales against each authorized observation.',
        'no_question_action':True, 'source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
    (root/'comparison.json').write_text(json.dumps(result,indent=2));backup(root)
    print(json.dumps({k:{key:val for key,val in v.items() if key not in ('auditor_reasons','channel_behavior')} for k,v in comparisons.items()},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    analyze(p.parse_args().root)
