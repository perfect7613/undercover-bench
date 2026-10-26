"""One frozen, full-constitution controlled non-game final check; no tuning."""
import argparse
import asyncio
import json

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.eval.benchmarks import BenchmarkConfig,register,run_benchmark
from tinker_cookbook.rl.message_env import EnvFromMessageEnv

from cookbook_distillation_env import DeliveryBenchmark,DeliveryEnv,DATA,load_cases,SYSTEM
from paper_character_v7 import RUN,HERE,BASE_RECORD,RENDERER,digest,metadata,read_jsonl


async def evaluate(condition):
    lock=json.loads((HERE/'FINAL_CHARACTER_EVAL_LOCK.json').read_text())
    if lock['cases_sha256']!=digest(DATA.read_text()) or lock['constitution_sha256']!=digest((HERE/'CONSTITUTION.md').read_text()):
        raise ValueError('frozen final evaluation inputs changed')
    checkpoint=lock['checkpoints'][condition]
    folder=RUN/'frozen_final_v8'/condition
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'manifest.json').write_text(json.dumps({**metadata(),'condition':condition,'checkpoint':checkpoint,
        'lock_sha256':digest(lock),'full_constitution_control':condition=='constitution',
        'no_checkpoint_selection_or_further_training_from_final_results':True,
        'sampling_seeds':'cookbook defaults, not explicitly paired'},indent=2))
    service=tinker.ServiceClient()
    client=await asyncio.to_thread(service.create_sampling_client,model_path=checkpoint)
    renderer=renderers.get_renderer(RENDERER,client.get_tokenizer())
    rows=[]
    try:
        for replicate in range(2):
            class FrozenBenchmark(DeliveryBenchmark):
                def __init__(self):
                    super().__init__('final',False)
                    self.name=f'auditor_final_v8_{condition}_replica_{replicate}'
                def make_envs(self,renderer,config):
                    environments=[]
                    for case in load_cases('final'):
                        env=DeliveryEnv(case,False)
                        if condition=='constitution':
                            original=env.initial_observation
                            async def initialized(env=env,original=original):
                                messages=await original()
                                messages[0]['content']=SYSTEM+'\n'+(HERE/'CONSTITUTION.md').read_text()
                                return messages
                            env.initial_observation=initialized
                        environments.append(EnvFromMessageEnv(renderer,env,failed_parse_reward=0,context_overflow_reward=0))
                    return environments
            builder=FrozenBenchmark();register(builder)
            # Single-replicate mode resumes completed examples instead of silently rerunning pass@k.
            await run_benchmark(builder.name,client,renderer,
                BenchmarkConfig(save_dir=str(folder),num_samples=1,temperature=.7,max_tokens=8192,
                                agent_concurrency=2,timeout_seconds=600))
            rows.extend({**r,'replicate':replicate} for r in read_jsonl(folder/builder.name/'trajectories.jsonl'))
        result={'condition':condition,'checkpoint':checkpoint,'episodes':len(rows),
                'correct':sum(r['reward']==1 and not r.get('error') for r in rows),
                'errors':sum(bool(r.get('error')) for r in rows),'metrics':{},
                'limitation':'small structurally templated non-game check; cannot by itself establish broad character learning'}
        for name in ['auditor_correct','work_complete','supported_report','false_report','missed_report','invalid_action']:
            result['metrics'][name]=sum(r.get('metrics',{}).get(name,0) for r in rows)
        (folder/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    finally:
        await asyncio.to_thread(lambda:service.close('success').result())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition',choices=('base','constitution','dpo','sft'))
    asyncio.run(evaluate(parser.parse_args().condition))
