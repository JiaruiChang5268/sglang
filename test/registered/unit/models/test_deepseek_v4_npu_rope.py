"""Regression tests for DSV4 NPU RoPE position handling."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.deepseek_v4_rope import ensure_npu_interleaved_rope_cache
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_freqs_cis(max_pos: int = 16, rope_dim: int = 8) -> torch.Tensor:
    positions = torch.arange(max_pos, dtype=torch.float32)
    inv_freq = torch.arange(1, rope_dim // 2 + 1, dtype=torch.float32) / 100.0
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


class TestDeepseekV4NpuRope(CustomTestCase):
    def test_same_shape_forward_uses_current_positions(self):
        freqs_cis = _make_freqs_cis()
        rotary_emb = torch.nn.Module()
        ensure_npu_interleaved_rope_cache(rotary_emb, freqs_cis, torch.float32)
        layer = SimpleNamespace(rotary_emb=rotary_emb, freqs_cis=freqs_cis)

        first_positions = torch.tensor([1, 2], dtype=torch.long)
        second_positions = torch.tensor([7, 9], dtype=torch.long)

        first_cos, _ = MQALayer._get_npu_rope_position_cache(
            layer, first_positions, torch.float32
        )
        second_cos, second_sin = MQALayer._get_npu_rope_position_cache(
            layer, second_positions, torch.float32
        )

        expected_cos = (
            freqs_cis.real.index_select(0, second_positions)
            .repeat_interleave(2, dim=-1)
            .view(2, 1, 1, -1)
        )
        expected_sin = (
            freqs_cis.imag.index_select(0, second_positions)
            .repeat_interleave(2, dim=-1)
            .view(2, 1, 1, -1)
        )

        self.assertFalse(torch.equal(first_cos, second_cos))
        self.assertTrue(torch.equal(second_cos, expected_cos))
        self.assertTrue(torch.equal(second_sin, expected_sin))
        self.assertFalse(hasattr(rotary_emb, "position_cos_layer_cache"))
        self.assertFalse(hasattr(rotary_emb, "position_sin_layer_cache"))


if __name__ == "__main__":
    unittest.main()
