"""Resumable single-replicate cookbook evaluations; never retry wrong answers."""
import argparse
import asyncio
import json
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.eval.benchmarks import BenchmarkConfig,register,run_benchmark

from cookbook_distillation_env import DeliveryBenchmark
from paper_character_v7 import RUN,BASE_RECORD,RENDERER,metadata,digest,read_jsonl,append
from retry_character_infrastructure import infrastructure_error


async def evaluate(condition,split):
    record=BASE_RECORD if condition in ('base','constitution') else RUN/condition/'checkpoint.json'
    checkpoint=json.loads(record.read_text())['sampler_path']
    folder=RUN/'eval'/condition/split
    folder.mkdir(parents=True,exist_ok=True)
    config={**metadata(),'condition':condition,'split':split,'checkpoint':checkpoint,
            'replicates':2,'protocol':'v7b_resumable_replicates','temperature':.7,
            'max_tokens':8192,'seeds':'cookbook default, not explicitly paired'}
    path=folder/'readiness_manifest.json'
    if path.exists() and json.loads(path.read_text())!=config:
        old=json.loads(path.read_text())
        if any(old.get(k)!=config[k] for k in ('condition','split','checkpoint','protocol')):
            raise ValueError('cannot resume a different checkpoint/protocol')
    path.write_text(json.dumps(config,indent=2))
    service=tinker.ServiceClient()
    client=await asyncio.to_thread(service.create_sampling_client,model_path=checkpoint)
    renderer=renderers.get_renderer(RENDERER,client.get_tokenizer())
    canonical=[]
    try:
        for replicate in range(2):
            builder=DeliveryBenchmark(split,condition=='constitution')
            canonical_name=builder.name
            builder.name+=f'_readiness_replica_{replicate}'
            register(builder)
            save_dir=folder/'replicates'/str(replicate)
            file=save_dir/builder.name/'trajectories.jsonl'
            # At most one explicit retry of an infrastructure failure per invocation.
            if file.exists():
                old=read_jsonl(file)
                errors=[r for r in old if infrastructure_error(r)]
                for error in errors:
                    append(folder/'transport_attempts.jsonl',{'replicate':replicate,'trajectory':error})
                if errors:
                    file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old if not infrastructure_error(r)))
            await run_benchmark(builder.name,client,renderer,
                BenchmarkConfig(save_dir=str(save_dir),num_samples=1,temperature=.7,max_tokens=8192,
                                agent_concurrency=2,timeout_seconds=600))
            rows=read_jsonl(file)
            for row in rows:
                canonical.append({**row,'benchmark':canonical_name,'replicate':replicate})
        destination=folder/canonical_name
        destination.mkdir(exist_ok=True)
        target=destination/'trajectories.jsonl'
        if target.exists() and not (destination/'previous_trajectories.jsonl').exists():
            (destination/'previous_trajectories.jsonl').write_bytes(target.read_bytes())
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in canonical))
        summary={'condition':condition,'split':split,'checkpoint':checkpoint,'episodes':len(canonical),
                 'correct':sum(r['reward']==1 and not r.get('error') for r in canonical),
                 'errors':sum(bool(r.get('error')) for r in canonical),'rows_sha256':digest(canonical),
                 'improvement_established':False,'cleared_for_rlcr':False}
        (folder/'readiness_result.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
    finally:
        await asyncio.to_thread(lambda:service.close('success').result())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition',choices=('base','constitution','dpo','sft'))
    parser.add_argument('--split',choices=('dev','final'),default='dev')
    args=parser.parse_args();asyncio.run(evaluate(args.condition,args.split))
