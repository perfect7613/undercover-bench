import unittest
import datasets
from paper_introspection_v7 import normalize_sft_messages


class IntrospectionFormattingTests(unittest.TestCase):
    def test_native_thinking_and_text_survive_homogeneous_arrow_roundtrip(self):
        messages=[{'role':'user','content':'Explain a commitment.'},
                  {'role':'assistant','content':[{'type':'thinking','thinking':'native emitted reasoning'},
                                                {'type':'text','text':'An evidence-sensitive answer.'}]}]
        normalized=normalize_sft_messages(messages)
        rows=datasets.Dataset.from_list([{'messages':normalized}])
        result=rows[0]['messages']
        self.assertEqual(result[0]['content'][0]['text'],messages[0]['content'])
        self.assertEqual(result[1]['content'][0]['thinking'],'native emitted reasoning')
        self.assertEqual(result[1]['content'][1]['text'],'An evidence-sensitive answer.')
        self.assertIsInstance(messages[0]['content'],str)

    def test_unexpected_modality_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_sft_messages([{'role':'assistant','content':[{'type':'image','image':'x'}]}])


if __name__=='__main__':unittest.main()
