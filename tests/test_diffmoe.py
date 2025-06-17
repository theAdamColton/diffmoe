import unittest

import torch

from diffmoe.diffmoe import DiffMoeMLP, masked_mean


class TestDiffMoe(unittest.TestCase):
    def test_fullgraph(self):
        # Padding shouldnt effect outputs whatsoever
        embed_dim = 8
        sequence_length = 32
        dtype = torch.float32
        mlp = DiffMoeMLP(embed_dim=embed_dim).to(dtype)
        mlp = torch.compile(mlp, fullgraph=True)
        x = torch.randn(sequence_length, embed_dim, dtype=dtype)
        outputs = mlp(x)

    def test_padding(self):
        # Padding shouldnt effect outputs whatsoever
        embed_dim = 8
        sequence_length = 100
        padding_length = 50
        dtype = torch.float32
        mlp = DiffMoeMLP(embed_dim=embed_dim).to(dtype)
        x = torch.randn(sequence_length, embed_dim, dtype=dtype)
        padding = torch.randn(padding_length, embed_dim, dtype=dtype)
        x_padded = torch.cat((x, padding))

        padding_mask = torch.zeros(sequence_length + padding_length, dtype=torch.bool)
        padding_mask[sequence_length:] = 1

        # The capacity should be scaled by the number of padding tokens,
        # so that there are the same number of activated (non-padding) tokens

        with torch.no_grad():
            mlp.training_capacity = 1.0
            y, *_ = mlp(x, padding_mask=padding_mask[:sequence_length])
            mlp.training_capacity = sequence_length / (sequence_length + padding_length)
            y_padded, *_ = mlp(x_padded, padding_mask=padding_mask)

        y_unpadded = y_padded[:sequence_length]

        self.assertTrue(torch.allclose(y, y_unpadded, rtol=0.01, atol=0.01))

        mlp = mlp.eval()
        with torch.no_grad():
            mlp.training_capacity = 1.0
            y, *_ = mlp(x, padding_mask=padding_mask[:sequence_length])
            y_padded, *_ = mlp(x_padded, padding_mask=padding_mask)

        y_unpadded = y_padded[:sequence_length]
        self.assertTrue(torch.allclose(y, y_unpadded, rtol=0.01, atol=0.01))

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
        num_steps = 500
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
            y_eval, *_ = mlp(x)

        self.assertTrue(torch.allclose(y, y_eval, rtol=0.1, atol=0.05))

    def test_masked_mean(self):
        x = torch.randn(1024, 64) * 5
        m = torch.randn(1024) > 0.5
        mean = x[m].mean()
        mean_hat = masked_mean(x, m)
        self.assertAlmostEqual(mean.item(), mean_hat.item())
