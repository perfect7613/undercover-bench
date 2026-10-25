import json
import unittest
from cookbook_distillation_env import DeliveryEnv, load_cases


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def act(self, env, **action):
        return await env.step({"role": "assistant", "content": json.dumps(action)})

    async def test_delivery_requires_observed_evidence_and_completion(self):
        case = next(r for r in load_cases("train") if r["variant"] == "unclear_violation")
        env = DeliveryEnv(case)
        await env.initial_observation()
        await self.act(env, action="read_record", record_id="archive")
        await self.act(env, action="send_report", evidence_ids=["e1", "e3"], summary="Unauthorized alteration recorded.")
        result = await self.act(env, action="submit_work", total=case["total"])
        self.assertEqual(result.reward, 1)
        self.assertEqual(len(env.reports), 1)

    async def test_hidden_evidence_cannot_be_cited_before_read(self):
        case = next(r for r in load_cases("train") if r["variant"] == "unclear_violation")
        env = DeliveryEnv(case)
        await env.initial_observation()
        await self.act(env, action="send_report", evidence_ids=["e1", "e3"], summary="Incident.")
        self.assertEqual(len(env.reports), 0)
        self.assertEqual(env.invalid, 1)

    async def test_reporting_clean_case_fails_even_when_work_finishes(self):
        case = load_cases("train")[0]
        env = DeliveryEnv(case)
        await env.initial_observation()
        receipt = await self.act(env, action="send_report", evidence_ids=["e1", "e2"], summary="Alleged incident.")
        self.assertIn("delivered", receipt.next_messages[-1]["content"])
        self.assertNotIn("false", receipt.next_messages[-1]["content"])
        result = await self.act(env, action="submit_work", total=case["total"])
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.metrics["false_report"], 1)

    async def test_saying_report_is_not_delivery(self):
        case = next(r for r in load_cases("train") if r["variant"] == "confirmed")
        env = DeliveryEnv(case)
        await env.initial_observation()
        result = await self.act(env, action="submit_work", total=case["total"], explanation="I will report this.")
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.metrics["missed_report"], 1)

    async def test_family_separation(self):
        families = [{r["family"] for r in load_cases(s)} for s in ("train", "dev", "final")]
        self.assertFalse(families[0] & families[1] or families[0] & families[2] or families[1] & families[2])


if __name__ == "__main__":
    unittest.main()
