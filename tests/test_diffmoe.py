import unittest

import torch

from diffmoe.diffmoe import DiffMoeMLP


class TestDiffMoe(unittest.TestCase):
    def test_train_matches_eval(self):
        embed_dim = 8
        dtype = torch.float64
        mlp = DiffMoeMLP(embed_dim=embed_dim).to(dtype)
        x = torch.randn(100, embed_dim, dtype=dtype)

        # Train only the capacity predictor for several steps,
        # so that the capacity predictor overfits and correctly outputs
        # the keep_mask. After this overfitting the capacity predictor
        # should exactly output the same exact experts for each token
        # as the 'training mode' batch pooled scores
        optim = torch.optim.Adam(mlp.parameters(), lr=1e-2)
        num_steps = 200
        for _ in range(num_steps):
            y, loss = mlp(x)
            loss.backward()
            optim.step()
            optim.zero_grad()

        # Test that the mlp has the same outputs
        # for training mode and for eval mode

        with torch.no_grad():
            y, loss = mlp(x)

        mlp = mlp.eval()
        with torch.no_grad():
            y_eval = mlp(x)

        self.assertTrue(torch.allclose(y, y_eval, rtol=0.1, atol=0.05))
