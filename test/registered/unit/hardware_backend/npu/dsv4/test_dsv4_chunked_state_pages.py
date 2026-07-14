"""CPU tests for DSV4 chunked-prefill state page masking."""

import unittest

from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    dsv4_prefill_state_page_zero_ranges,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4ChunkedStatePages(CustomTestCase):
    def test_c4_preserves_previous_chunk_overlap_page(self):
        self.assertEqual(
            dsv4_prefill_state_page_zero_ranges(
                prefix_len=4096,
                first_tail_page=63,
                page_size=128,
                compress_ratio=4,
            ),
            ((0, 31), (32, 63)),
        )

    def test_c4_preserves_overlap_spanning_two_pages(self):
        self.assertEqual(
            dsv4_prefill_state_page_zero_ranges(
                prefix_len=129,
                first_tail_page=5,
                page_size=128,
                compress_ratio=4,
            ),
            ((2, 5),),
        )

    def test_c4_first_chunk_has_no_previous_overlap(self):
        self.assertEqual(
            dsv4_prefill_state_page_zero_ranges(
                prefix_len=0,
                first_tail_page=31,
                page_size=128,
                compress_ratio=4,
            ),
            ((0, 31),),
        )

    def test_c128_keeps_only_final_tail_pages(self):
        self.assertEqual(
            dsv4_prefill_state_page_zero_ranges(
                prefix_len=4096,
                first_tail_page=63,
                page_size=128,
                compress_ratio=128,
            ),
            ((0, 63),),
        )


if __name__ == "__main__":
    unittest.main()
