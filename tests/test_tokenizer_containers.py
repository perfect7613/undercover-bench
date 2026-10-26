"""Native tokenizer return types must preserve the exact sampled token prefix."""
import unittest
from types import SimpleNamespace

from undercover_bench.rlcr import RLCRInputError
from undercover_bench.torch_backend import normalize_token_ids


class TokenContainerTests(unittest.TestCase):
    def test_native_containers_preserve_all_ids(self):
        for value in ([1, 2, 3], [[1, 2, 3]], {"input_ids": [[1, 2, 3]]},
                      SimpleNamespace(ids=[1, 2, 3]), [SimpleNamespace(ids=[1, 2, 3])]):
            self.assertEqual(normalize_token_ids(value), [1, 2, 3])

    def test_multiple_sequences_or_noninteger_tokens_are_rejected(self):
        for value in ([[1], [2]], [], [True], [1.0], [-1], [SimpleNamespace(ids=[1]), SimpleNamespace(ids=[2])]):
            with self.assertRaises(RLCRInputError):
                normalize_token_ids(value)

    def test_real_tokenizers_encoding(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        tokenizer = Tokenizer(WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]"))
        encoding = tokenizer.encode("hello")
        self.assertEqual(normalize_token_ids([encoding]), [0])
