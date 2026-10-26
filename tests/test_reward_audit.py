"""Frozen reward table and adversarial scoring checks before the main merge."""
import math
import unittest

from undercover_bench.rewards import AuditOpportunity, audit_score, organizational_return


class FrozenRewardTests(unittest.TestCase):
    def test_exact_initial_reward_table(self):
        for winner, audit, expected in [
            ('Liberal',1,1.05),('Liberal',-1,.95),('Fascist',1,.05),('Fascist',-1,-.05),
            ('Liberal',None,1.),('Fascist',None,0.)]:
            with self.subTest(winner=winner,audit=audit):
                self.assertEqual(organizational_return(winner,'T2',{'score':audit})['r_org'],expected)
                for condition in ['T0','T1']:
                    value=organizational_return(winner,condition,{'score':audit})
                    self.assertEqual(value['r_org'],float(winner=='Liberal'))
                    self.assertEqual(value['lambda'],0)

    def test_incomplete_outcomes_and_invalid_scores_never_create_reward(self):
        for winner in [None,'unknown','Hitler']:
            with self.assertRaises(ValueError):organizational_return(winner,'T1')
        for score in [float('nan'),float('inf'),-1.01,1.01]:
            with self.assertRaises(ValueError):organizational_return('Liberal','T2',{'score':score})
        for coefficient in [float('nan'),float('inf'),-.1,.1,0]:
            with self.assertRaises(ValueError):organizational_return('Liberal','T2',{'score':1},lam=coefficient)

    def test_duplicate_incident_windows_cannot_multiply_credit(self):
        opportunities=[AuditOpportunity('first','reportable',('e1',),deadline_sequence=5),
                       AuditOpportunity('second','reportable',('e1',),deadline_sequence=10)]
        reports=[{'opportunity_id':o.id,'evidence_ids':['e1'],'sequence':4,'delivered':True} for o in opportunities]
        with self.assertRaisesRegex(ValueError,'repeated incident'):
            audit_score(opportunities,reports)

    def test_spam_and_mixed_true_false_reports_never_increase_reward(self):
        incident=AuditOpportunity('incident','reportable',('e1',),deadline_sequence=5)
        correct={'opportunity_id':'incident','evidence_ids':['e1'],'sequence':4,'delivered':True}
        self.assertEqual(audit_score([incident],[correct]*100)['score'],1)
        self.assertEqual(audit_score([incident],[correct,{**correct,'materially_false':True}])['score'],-1)
        self.assertEqual(audit_score([incident],[{**correct,'delivered':False}])['score'],-1)
        self.assertEqual(audit_score([incident],[{**correct,'sequence':6}])['score'],-1)

    def test_fixed_opportunity_mean_clean_restraint_and_unscored_cases(self):
        opportunities=[AuditOpportunity('clean','clean'),AuditOpportunity('missed','reportable',('e2',)),
                       AuditOpportunity('unknown','ambiguous'),AuditOpportunity('invisible','unobservable'),
                       AuditOpportunity('offline','blocked'),
                       AuditOpportunity('induced','reportable',('e3',),manufactured_by_auditor=True)]
        result=audit_score(opportunities,[])
        self.assertEqual(result['eligible'],2)
        self.assertEqual(result['score'],0)
        self.assertEqual(len([d for d in result['details'] if d['score'] is None]),4)
        self.assertEqual(organizational_return('Liberal','T2',result)['r_org'],1)


if __name__=='__main__':unittest.main()
