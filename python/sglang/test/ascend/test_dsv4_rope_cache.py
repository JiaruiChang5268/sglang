import pytest
import torch

from sglang.srt.layers.deepseek_v4_rope import (
    ensure_npu_interleaved_rope_cache,
    get_fused_compressor_rope_cos_sin,
    get_npu_interleaved_rope_cos_sin,
    npu_partial_rotary_mul_inplace,
)


class _CacheOwner:
    pass


def _make_freqs_cis(max_pos: int = 16, rope_dim: int = 8) -> torch.Tensor:
    t = torch.arange(max_pos, dtype=torch.float32)
    inv_freq = torch.arange(1, rope_dim // 2 + 1, dtype=torch.float32) / 100.0
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def test_npu_interleaved_rope_cache_matches_repeat_interleave():
    freqs_cis = _make_freqs_cis()
    owner = _CacheOwner()
    positions = torch.tensor([0, 3, 7, 3], dtype=torch.long)

    cos_cache, sin_cache = ensure_npu_interleaved_rope_cache(
        owner, freqs_cis, torch.bfloat16
    )
    expected_cos = freqs_cis.real.repeat_interleave(2, dim=-1).to(torch.bfloat16)
    expected_sin = freqs_cis.imag.repeat_interleave(2, dim=-1).to(torch.bfloat16)
    assert torch.equal(cos_cache, expected_cos)
    assert torch.equal(sin_cache, expected_sin)

    cos4, sin4 = get_npu_interleaved_rope_cos_sin(
        owner, freqs_cis, positions, torch.bfloat16, view_4d=True
    )
    assert cos4.shape == (positions.numel(), 1, 1, freqs_cis.shape[-1] * 2)
    assert sin4.shape == cos4.shape
    assert torch.equal(cos4.view(positions.numel(), -1), expected_cos[positions])
    assert torch.equal(sin4.view(positions.numel(), -1), expected_sin[positions])

    _, inverse_sin4 = get_npu_interleaved_rope_cos_sin(
        owner, freqs_cis, positions, torch.bfloat16, view_4d=True, inverse=True
    )
    assert torch.equal(
        inverse_sin4.view(positions.numel(), -1), -expected_sin[positions]
    )


def test_npu_interleaved_rope_no_build_path_requires_initialized_cache():
    freqs_cis = _make_freqs_cis()
    owner = _CacheOwner()
    positions = torch.tensor([1, 4], dtype=torch.long)

    with pytest.raises(RuntimeError, match="cache is missing"):
        get_npu_interleaved_rope_cos_sin(
            owner,
            freqs_cis,
            positions,
            torch.bfloat16,
            view_4d=True,
            allow_build=False,
        )

    ensure_npu_interleaved_rope_cache(owner, freqs_cis, torch.bfloat16)
    cos4, sin4 = get_npu_interleaved_rope_cos_sin(
        owner,
        freqs_cis,
        positions,
        torch.bfloat16,
        view_4d=True,
        allow_build=False,
    )
    assert cos4.shape == (positions.numel(), 1, 1, freqs_cis.shape[-1] * 2)
    assert sin4.shape == cos4.shape


def test_npu_interleaved_rope_can_return_runtime_dtype_from_fp32_cache():
    freqs_cis = _make_freqs_cis()
    owner = _CacheOwner()
    positions = torch.tensor([1, 4], dtype=torch.long)

    ensure_npu_interleaved_rope_cache(owner, freqs_cis, torch.float32)
    cos4, sin4 = get_npu_interleaved_rope_cos_sin(
        owner,
        freqs_cis,
        positions,
        torch.bfloat16,
        view_4d=True,
        allow_build=False,
        cache_dtype=torch.float32,
    )

    expected_cos = freqs_cis.real.repeat_interleave(2, dim=-1).to(torch.bfloat16)
    expected_sin = freqs_cis.imag.repeat_interleave(2, dim=-1).to(torch.bfloat16)
    assert cos4.dtype == torch.bfloat16
    assert sin4.dtype == torch.bfloat16
    assert torch.equal(cos4.view(positions.numel(), -1), expected_cos[positions])
    assert torch.equal(sin4.view(positions.numel(), -1), expected_sin[positions])


def test_fused_compressor_rope_cos_sin_matches_old_path():
    freqs_cis = _make_freqs_cis()
    owner = _CacheOwner()
    positions = torch.tensor([2, 5, 5, 8], dtype=torch.long)

    cos, sin = get_fused_compressor_rope_cos_sin(
        freqs_cis, positions, dtype=torch.float32, cache_owner=owner
    )
    expected_cos = (
        freqs_cis.real.contiguous()
        .index_select(0, positions)
        .repeat_interleave(2, dim=-1)
        .to(torch.float32)
    )
    expected_sin = (
        freqs_cis.imag.contiguous()
        .index_select(0, positions)
        .repeat_interleave(2, dim=-1)
        .to(torch.float32)
    )
    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, expected_sin)


def _apply_interleaved_rope_ref(
    x: torch.Tensor, cos4: torch.Tensor, sin4: torch.Tensor, qk_nope: int
) -> torch.Tensor:
    out = x.clone()
    rope_dim = cos4.shape[-1]
    rope = out[..., qk_nope : qk_nope + rope_dim]
    cos = cos4.view(cos4.shape[0], rope_dim)
    sin = sin4.view(sin4.shape[0], rope_dim)
    x_even = rope[..., 0::2].clone()
    x_odd = rope[..., 1::2].clone()
    cos_half = cos[:, None, 0::2]
    sin_half = sin[:, None, 0::2]
    rope[..., 0::2] = x_even * cos_half - x_odd * sin_half
    rope[..., 1::2] = x_odd * cos_half + x_even * sin_half
    return out


def test_npu_partial_rotary_mul_inplace_matches_reference_on_npu():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")

    device = torch.device("npu")
    freqs_cis = _make_freqs_cis(max_pos=8, rope_dim=8).to(device)
    positions = torch.tensor([0, 2, 7], dtype=torch.long, device=device)
    qk_nope = 4
    q = torch.randn(3, 2, qk_nope + 8, dtype=torch.bfloat16, device=device)
    kv = torch.randn(3, 1, qk_nope + 8, dtype=torch.bfloat16, device=device)
    cos4, sin4 = get_npu_interleaved_rope_cos_sin(
        None, freqs_cis, positions, q.dtype, view_4d=True
    )

    q_ref = _apply_interleaved_rope_ref(q, cos4, sin4, qk_nope)
    kv_ref = _apply_interleaved_rope_ref(kv, cos4, sin4, qk_nope)
    npu_partial_rotary_mul_inplace(q, kv, cos4, sin4, qk_nope=qk_nope)

    torch.npu.synchronize()
    assert torch.allclose(q, q_ref, atol=0, rtol=0)
    assert torch.allclose(kv, kv_ref, atol=0, rtol=0)
