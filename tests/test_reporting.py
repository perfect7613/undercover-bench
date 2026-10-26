import json
import tempfile
import unittest
from pathlib import Path

from undercover_bench.reporting import OversightInbox,receipt_digest


class ReportingTests(unittest.TestCase):
    def test_persisted_receipt_is_separate_and_does_not_claim_human_read(self):
        with tempfile.TemporaryDirectory() as directory:
            box=OversightInbox(directory,'episode-1')
            first=box.report_to_human('Aster',['e1'],{'e1'},'Observed proposal; no acceptance observed.')
            second=box.report_to_human('Aster',['e1'],{'e1'},'Duplicate submission; no extra scoring credit.')
            rows=[json.loads(x) for x in box.path.read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertTrue(first['delivered'])
            self.assertFalse(rows[0]['human_read_acknowledged'])
            self.assertEqual(first['receipt_sha256'],receipt_digest(rows[0]))
            self.assertNotEqual(first['receipt_id'],second['receipt_id'])
            self.assertEqual(rows[0]['verification'],'pending_independent_review')

    def test_invented_evidence_and_unsafe_episode_names_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):OversightInbox(directory,'../outside')
            box=OversightInbox(directory,'episode-1')
            with self.assertRaises(ValueError):box.report_to_human('Aster',['invented'],{'e1'})
            self.assertFalse(box.path.exists())


if __name__=='__main__':unittest.main()
