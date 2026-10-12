import unittest
import torch
from smoothquant.fake_quant import (
    quantize_weight_per_channel_absmax,
    quantize_weight_per_tensor_absmax,
    quantize_activation_per_token_absmax,
    quantize_activation_per_tensor_absmax,
    W8A8Linear,
)


class TestFakeQuant(unittest.TestCase):
    def test_weight_per_channel_absmax_clamping(self):
        for n_bits in [4, 8]:
            q_max = 2 ** (n_bits - 1) - 1
            # Create a tensor where division and rounding could hit boundary
            # e.g., shape (4, 16)
            w = torch.randn(4, 16, dtype=torch.float32)
            # Inject extreme boundary values
            w[0, 0] = -1e5
            w[0, 1] = 1e5
            scales = w.abs().max(dim=-1, keepdim=True)[0]
            scales.clamp_(min=1e-5).div_(q_max)

            w_quant = quantize_weight_per_channel_absmax(w.clone(), n_bits=n_bits)
            int_vals = torch.round(w_quant / scales)
            self.assertTrue(torch.all(int_vals >= -q_max))
            self.assertTrue(torch.all(int_vals <= q_max))

    def test_weight_per_tensor_absmax_clamping(self):
        for n_bits in [4, 8]:
            q_max = 2 ** (n_bits - 1) - 1
            w = torch.randn(8, 8, dtype=torch.float32)
            w[0, 0] = -1e4
            w[0, 1] = 1e4
            scale = w.abs().max()
            scale.clamp_(min=1e-5).div_(q_max)

            w_quant = quantize_weight_per_tensor_absmax(w.clone(), n_bits=n_bits)
            int_vals = torch.round(w_quant / scale)
            self.assertTrue(torch.all(int_vals >= -q_max))
            self.assertTrue(torch.all(int_vals <= q_max))

    def test_activation_per_token_absmax_clamping(self):
        for n_bits in [4, 8]:
            q_max = 2 ** (n_bits - 1) - 1
            t = torch.randn(2, 5, 10, dtype=torch.float32)
            t[0, 0, 0] = -1e5
            t[1, 2, 3] = 1e5
            scales = t.abs().max(dim=-1, keepdim=True)[0]
            scales.clamp_(min=1e-5).div_(q_max)

            t_quant = quantize_activation_per_token_absmax(t.clone(), n_bits=n_bits)
            int_vals = torch.round(t_quant / scales)
            self.assertTrue(torch.all(int_vals >= -q_max))
            self.assertTrue(torch.all(int_vals <= q_max))

    def test_activation_per_tensor_absmax_clamping(self):
        for n_bits in [4, 8]:
            q_max = 2 ** (n_bits - 1) - 1
            t = torch.randn(2, 4, 8, dtype=torch.float32)
            t[0, 0, 0] = -1e4
            t[1, 1, 1] = 1e4
            scale = t.abs().max()
            scale.clamp_(min=1e-5).div_(q_max)

            t_quant = quantize_activation_per_tensor_absmax(t.clone(), n_bits=n_bits)
            int_vals = torch.round(t_quant / scale)
            self.assertTrue(torch.all(int_vals >= -q_max))
            self.assertTrue(torch.all(int_vals <= q_max))

    def test_boundary_values_exact(self):
        # Directly test boundary edge case where round without clamp would exceed q_max
        # For instance, in float16 where division precision can cause rounding over q_max
        q_max = 127
        for dtype in [torch.float32, torch.float16]:
            w = torch.randn(50, 50, dtype=dtype)
            scales = w.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5).div(q_max)
            quantized = quantize_weight_per_channel_absmax(w.clone(), n_bits=8)
            int_vals = torch.round(quantized / scales)
            self.assertGreaterEqual(int_vals.min().item(), -q_max)
            self.assertLessEqual(int_vals.max().item(), q_max)

    def test_explicit_out_of_bounds_clamped(self):
        # Verify that if floating-point division precision rounds intermediate quotients
        # to +/- 128 (e.g. float16 division edge case v=0.0001220703125), it is strictly
        # clamped to +/- 127 in all 4 quantization functions.
        q_max = 127
        v = 0.0001220703125
        t_fp16 = torch.tensor([[v, -v]], dtype=torch.float16)
        scale = t_fp16.abs().max().clamp(min=1e-5).div(q_max)

        # Confirm that without clamp, rounding produces out-of-bounds +/- 128
        unclamped_int = torch.round(t_fp16.clone().div_(scale).round_().mul_(scale) / scale)
        self.assertEqual(unclamped_int.max().item(), 128.0)
        self.assertEqual(unclamped_int.min().item(), -128.0)

        # Verify all four quantization functions strictly clamp within [-q_max, q_max]
        quant_fns = [
            quantize_weight_per_channel_absmax,
            quantize_weight_per_tensor_absmax,
            quantize_activation_per_token_absmax,
            quantize_activation_per_tensor_absmax,
        ]
        for fn in quant_fns:
            out = fn(t_fp16.clone(), n_bits=8)
            int_vals = torch.round(out / scale)
            self.assertGreaterEqual(int_vals.min().item(), -q_max)
            self.assertLessEqual(int_vals.max().item(), q_max)
            # Verify cast to int8 does not overflow
            int8_vals = int_vals.to(torch.int8)
            self.assertEqual(int8_vals.max().item(), q_max)
            self.assertEqual(int8_vals.min().item(), -q_max)

    def test_w8a8_linear_from_float(self):
        linear = torch.nn.Linear(16, 32)
        w8a8 = W8A8Linear.from_float(linear, weight_quant="per_channel", act_quant="per_token")
        x = torch.randn(2, 8, 16)
        out = w8a8(x)
        self.assertEqual(out.shape, (2, 8, 32))


if __name__ == "__main__":
    unittest.main()
