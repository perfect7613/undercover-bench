"""Retry only recorded transport failures, preserving all behavioral outcomes."""
import argparse
import asyncio
import json
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.eval.benchmarks import BenchmarkConfig, register, run_benchmark
from tinker_cookbook.rl.message_env import EnvFromMessageEnv

from cookbook_distillation_env import DeliveryBenchmark, DeliveryEnv, load_cases
from paper_character_v7 import RUN, RENDERER, read_jsonl, digest, append, metadata
from run_paper_pipeline_v7 import check_behavior

INFRASTRUCTURE_ERRORS = ('response expired', 'connection reset', 'connection error', 'timed out', 'timeout (')


def infrastructure_error(row):
    text = str(row.get('error') or '').lower()
    return any(marker in text for marker in INFRASTRUCTURE_ERRORS)


async def retry(condition):
    folder = RUN / 'eval' / condition / 'dev'
    canonical = folder / 'auditor_delivery_v7b_dev/trajectories.jsonl'
    rows = read_jsonl(canonical)
    if len(rows) != 16:
        raise ValueError('expected the original 16 trajectories; refusing to guess missing outcomes')
    candidates = {DeliveryEnv(c).example_id:c for c in load_cases('dev')}
    original = folder / 'original_with_infrastructure_errors.jsonl'
    if not original.exists():
        original.write_bytes(canonical.read_bytes())
    checkpoint = json.loads((RUN / condition / 'checkpoint.json').read_text())
    service = tinker.ServiceClient()
    client = await asyncio.to_thread(service.create_sampling_client, model_path=checkpoint['sampler_path'])
    renderer = renderers.get_renderer(RENDERER, client.get_tokenizer())
    semaphore = asyncio.Semaphore(2)
    async def one(index, row):
        if not infrastructure_error(row):
            return
        if row['example_id'] not in candidates:
            raise ValueError('recorded example does not match frozen case hash')
        case = candidates[row['example_id']]
        class OneCase(DeliveryBenchmark):
            def __init__(self):
                super().__init__('dev', False)
                self.name = f'auditor_delivery_v7b_dev_infra_retry_{index}'
            def make_envs(self, renderer, config):
                return [EnvFromMessageEnv(renderer, DeliveryEnv(case), failed_parse_reward=0,context_overflow_reward=0)]
        builder = OneCase()
        register(builder)
        attempt_folder = folder / 'infrastructure_retry' / str(index)
        async with semaphore:
            result = await run_benchmark(builder.name,client,renderer,
                BenchmarkConfig(save_dir=str(attempt_folder),num_samples=1,
                                temperature=.7,max_tokens=8192,agent_concurrency=1,timeout_seconds=600))
        replacement = read_jsonl(attempt_folder / builder.name / 'trajectories.jsonl')
        if len(replacement) != 1:
            raise ValueError('retry produced unexpected number of trajectories')
        attempt = replacement[0]
        append(folder / 'infrastructure_attempts.jsonl', {'original_position':index,'prior':row,
              'replacement':attempt,'checkpoint':checkpoint['sampler_path'],**metadata()})
        # Preserve a completed wrong answer exactly; it must not be retried for a higher score.
        if not attempt.get('error'):
            rows[index] = {**attempt,'idx':row['idx'],'benchmark':row['benchmark']}
        print(json.dumps({'position':index,'correct':attempt['reward'],'error':bool(attempt.get('error'))}),flush=True)
    try:
        await asyncio.gather(*(one(i,row) for i,row in enumerate(rows)))
        canonical.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        summary={'checkpoint':checkpoint['sampler_path'],'rows_sha256':digest(rows),'episodes':len(rows),
                 'correct':sum(r['reward']==1 and not r.get('error') for r in rows),
                 'errors':sum(bool(r.get('error')) for r in rows),'behavioral_outcomes_retried':False}
        (folder / 'infrastructure_recovery.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
        check_behavior(condition)
    finally:
        await asyncio.to_thread(lambda:service.close('success').result())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition',choices=('dpo','sft'))
    args=parser.parse_args()
    asyncio.run(retry(args.condition))
