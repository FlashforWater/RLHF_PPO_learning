import unittest

import torch

from rlhf.advantage import compute_gae


class ComputeGaeTest(unittest.TestCase):
    def test_mask_stops_bootstrap_after_eos(self):
        rewards = torch.tensor([[0.0, 1.0, 0.0]])
        values = torch.tensor([[0.5, 0.25, 10.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=1.0, mask=mask)

        expected_advantages = torch.tensor([[0.5, 0.75, 0.0]])
        expected_returns = torch.tensor([[1.0, 1.0, 10.0]])
        self.assertTrue(torch.allclose(advantages, expected_advantages))
        self.assertTrue(torch.allclose(returns, expected_returns))


if __name__ == "__main__":
    unittest.main()
