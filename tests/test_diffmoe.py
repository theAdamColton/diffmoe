import unittest

import torch

from diffmoe.diffmoe import DiffMoeMLP


class TestDiffMoe(unittest.TestCase):
    def test_uncompiled(self):
        embed_dim = 8
        mlp = DiffMoeMLP(embed_dim=embed_dim)
        x = torch.randn(100, embed_dim)
        with torch.no_grad():
            y, loss = mlp(x)

        mlp = mlp.eval()
        with torch.no_grad():
            y_eval = mlp(x)

        self.assertTrue(torch.allclose(y, y_eval))
