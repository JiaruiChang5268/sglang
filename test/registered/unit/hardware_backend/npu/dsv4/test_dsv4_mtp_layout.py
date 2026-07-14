"""CPU tests for request-aware DSV4-NPU MTP cache locations."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_mtp_layout import (
    build_dsv4_topk1_step_cache_locs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_pool(rows: int = 16, cols: int = 256):
    def table(width=cols):
        return torch.zeros((rows, width), dtype=torch.int32)

    return SimpleNamespace(
        req_to_token=table(),
        req_to_token_swa=table(),
        req_to_token_c4=table(max(1, cols // 4)),
        req_to_token_c128=table(max(1, cols // 128)),
        req_to_token_c4_state=table(),
        req_to_token_c128_state=table(),
    )


def _populate(pool, req_indices, seq_lens, num_steps):
    dense_full = []
    for req_idx, seq_len in zip(req_indices, seq_lens):
        for step_id in range(num_steps):
            pos = seq_len + step_id
            full = 100_000 + req_idx * 1_000 + pos
            pool.req_to_token[req_idx, pos] = full
            pool.req_to_token_swa[req_idx, pos] = full + 1_000_000
            pool.req_to_token_c4_state[req_idx, pos] = full + 2_000_000
            pool.req_to_token_c128_state[req_idx, pos] = full + 3_000_000
            if (pos + 1) % 4 == 0:
                pool.req_to_token_c4[req_idx, pos // 4] = full + 4_000_000
            if (pos + 1) % 128 == 0:
                pool.req_to_token_c128[req_idx, pos // 128] = full + 5_000_000
            dense_full.append(full)
    return torch.tensor(dense_full, dtype=torch.int64)


class TestDSV4MTPLayout(CustomTestCase):
    def test_request_major_layout_for_batch_sizes(self):
        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                pool = _make_pool()
                req_indices = [1, 3, 5, 7][:batch_size]
                seq_lens = [10, 21, 34, 47][:batch_size]
                dense = _populate(pool, req_indices, seq_lens, num_steps=2)

                for step_id in (0, 1):
                    locs = build_dsv4_topk1_step_cache_locs(
                        dense_full_locs=dense,
                        req_pool_indices=torch.tensor(req_indices),
                        seq_lens=torch.tensor(seq_lens),
                        req_to_token_pool=pool,
                        step_id=step_id,
                        num_steps=2,
                        padded_batch_size=batch_size,
                        validate=True,
                    )
                    expected = dense.view(batch_size, 2)[:, step_id].tolist()
                    self.assertEqual(locs.out_full_loc.tolist(), expected)
                    self.assertEqual(locs.real_batch_size, batch_size)

    def test_six_slot_reserve_bundle_does_not_define_step_layout(self):
        pool = _make_pool()
        req_indices = torch.tensor([1, 2])
        seq_lens = torch.tensor([10, 20])
        dense = torch.tensor([100, 101, 200, 201], dtype=torch.int64)
        for req_idx, seq_len, values in (
            (1, 10, (100, 101)),
            (2, 20, (200, 201)),
        ):
            for step_id, full in enumerate(values):
                pos = seq_len + step_id
                pool.req_to_token[req_idx, pos] = full
                pool.req_to_token_swa[req_idx, pos] = full + 1_000
                pool.req_to_token_c4_state[req_idx, pos] = full + 2_000
                pool.req_to_token_c128_state[req_idx, pos] = full + 3_000

        # The old code sliced the first bs*steps entries from this request-major
        # six-slot reservation, producing req0's third slot for req1.
        reserve_bundle = torch.tensor(
            [100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205]
        )
        old_step0 = reserve_bundle[:4].view(2, 1, 2).permute(2, 0, 1)[0].view(-1)
        self.assertEqual(old_step0.tolist(), [100, 102])

        locs = build_dsv4_topk1_step_cache_locs(
            dense_full_locs=dense,
            req_pool_indices=req_indices,
            seq_lens=seq_lens,
            req_to_token_pool=pool,
            step_id=0,
            num_steps=2,
            padded_batch_size=2,
            validate=True,
        )
        self.assertEqual(locs.out_full_loc.tolist(), [100, 200])

        # A reused reservation can allocate slots for req0 only. The dense
        # request-table view remains complete and is therefore the stable source.
        variable_reserve_bundle = reserve_bundle[:6]
        self.assertNotEqual(variable_reserve_bundle.numel(), dense.numel())
        self.assertEqual(locs.out_full_loc.tolist(), [100, 200])

    def test_graph_padding_uses_dummy_slot_zero(self):
        pool = _make_pool()
        req_indices = [1, 4]
        seq_lens = [10, 20]
        dense = _populate(pool, req_indices, seq_lens, num_steps=2)

        locs = build_dsv4_topk1_step_cache_locs(
            dense_full_locs=dense,
            req_pool_indices=torch.tensor([1, 4, 0, 0]),
            seq_lens=torch.tensor([10, 20, 1, 1]),
            req_to_token_pool=pool,
            step_id=1,
            num_steps=2,
            padded_batch_size=4,
            validate=True,
        )

        self.assertEqual(locs.real_batch_size, 2)
        self.assertEqual(locs.padded_batch_size, 4)
        self.assertEqual(locs.out_full_loc[-2:].tolist(), [0, 0])
        self.assertEqual(locs.out_swa_loc[-2:].tolist(), [0, 0])
        self.assertEqual(locs.out_c4_state_loc[-2:].tolist(), [0, 0])
        self.assertEqual(locs.out_c128_state_loc[-2:].tolist(), [0, 0])

    def test_compressed_locations_follow_per_request_boundaries(self):
        pool = _make_pool()
        req_indices = [1, 2, 3]
        seq_lens = [3, 127, 10]
        dense = _populate(pool, req_indices, seq_lens, num_steps=2)

        step0 = build_dsv4_topk1_step_cache_locs(
            dense_full_locs=dense,
            req_pool_indices=torch.tensor(req_indices),
            seq_lens=torch.tensor(seq_lens),
            req_to_token_pool=pool,
            step_id=0,
            num_steps=2,
            padded_batch_size=3,
            validate=True,
        )
        self.assertEqual(
            step0.out_c4_loc.tolist(),
            [
                int(pool.req_to_token_c4[1, 0]),
                int(pool.req_to_token_c4[2, 31]),
            ],
        )
        self.assertEqual(
            step0.out_c128_loc.tolist(),
            [int(pool.req_to_token_c128[2, 0])],
        )

        step1 = build_dsv4_topk1_step_cache_locs(
            dense_full_locs=dense,
            req_pool_indices=torch.tensor(req_indices),
            seq_lens=torch.tensor(seq_lens),
            req_to_token_pool=pool,
            step_id=1,
            num_steps=2,
            padded_batch_size=3,
            validate=True,
        )
        self.assertEqual(step1.out_c4_loc.numel(), 0)
        self.assertEqual(step1.out_c128_loc.numel(), 0)

    def test_debug_invariant_reports_first_full_loc_mismatch(self):
        pool = _make_pool()
        req_indices = [1, 2]
        seq_lens = [10, 20]
        dense = _populate(pool, req_indices, seq_lens, num_steps=2)
        dense[2] += 999

        with self.assertRaisesRegex(
            RuntimeError, "full cache-loc invariant failed.*batch_row=1"
        ):
            build_dsv4_topk1_step_cache_locs(
                dense_full_locs=dense,
                req_pool_indices=torch.tensor(req_indices),
                seq_lens=torch.tensor(seq_lens),
                req_to_token_pool=pool,
                step_id=0,
                num_steps=2,
                padded_batch_size=2,
                validate=True,
            )

    def test_rejects_inconsistent_dense_stride_and_graph_batch(self):
        pool = _make_pool()
        common = dict(
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([10, 20]),
            req_to_token_pool=pool,
            step_id=0,
            num_steps=2,
        )
        with self.assertRaisesRegex(ValueError, "not request-major"):
            build_dsv4_topk1_step_cache_locs(
                dense_full_locs=torch.tensor([1, 2, 3]),
                padded_batch_size=2,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "smaller than real_batch_size"):
            build_dsv4_topk1_step_cache_locs(
                dense_full_locs=torch.tensor([1, 2, 3, 4]),
                padded_batch_size=1,
                **common,
            )


if __name__ == "__main__":
    unittest.main()
