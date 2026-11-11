import unittest

import torch

from smoothquant.fake_quant import W8A8Linear


class W8A8LinearTest(unittest.TestCase):
    def test_transposed_batch_and_sequence(self):
        for mode in ("per_token", "per_tensor"):
            with self.subTest(act_quant=mode):
                linear = W8A8Linear.from_float(torch.nn.Linear(8, 5), act_quant=mode)
                x = torch.linspace(-2, 3, 96).reshape(3, 4, 8).transpose(0, 1)
                expected = linear(x.contiguous())
                torch.testing.assert_close(linear(x), expected)
