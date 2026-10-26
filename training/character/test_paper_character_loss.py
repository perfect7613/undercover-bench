import math
import unittest
import torch
from paper_character_loss import character_loss


class LossTests(unittest.TestCase):
    def test_reference_equality_is_log_two_plus_chosen_nll(self):
        p = [torch.tensor([-100., -1., -3.], requires_grad=True),
             torch.tensor([-100., -2., -2.], requires_grad=True)]
        mask = [torch.tensor([0., 1., 1.])]*2
        loss, metrics = character_loss(p, [v.detach().clone() for v in p], mask)
        self.assertAlmostEqual(loss.item(), math.log(2) + .1*2, places=6)
        self.assertEqual(metrics["squared_logratio_penalty"], 0)
        loss.backward()
        self.assertEqual(p[0].grad[0].item(), 0)
        self.assertLess(p[0].grad[1].item(), 0)
        self.assertGreater(p[1].grad[1].item(), 0)

    def test_penalty_averages_each_response_before_batch(self):
        p = [torch.tensor([-1., -1.]), torch.tensor([-2., -2., -2.])]
        r = [torch.tensor([-2., -2.]), torch.tensor([-4., -4., -4.])]
        _, metrics = character_loss(p, r, [torch.ones(2), torch.ones(3)])
        self.assertAlmostEqual(metrics["squared_logratio_penalty"], (1+4)/2)
        self.assertAlmostEqual(metrics["chosen_nll"], 1)

    def test_reject_normalized_masks(self):
        p = [torch.tensor([-1., -1.])]*2
        with self.assertRaises(ValueError):
            character_loss(p, p, [torch.tensor([.5, .5])]*2)

    def test_masked_values_do_not_affect_loss(self):
        p = [torch.tensor([float('nan'), -1.])]*2
        loss, _ = character_loss(p, p, [torch.tensor([0., 1.])]*2)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
