import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_utils import (
    build_dflash_verify_target_probs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashVerifyTargetProbs(CustomTestCase):
    def test_top_p_uses_torch_fallback_without_sgl_kernel(self):
        logits = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        sampling_info = SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=True,
            temperatures=torch.ones((2, 1)),
            top_ps=torch.tensor([0.7, 0.8]),
        )

        with patch(
            "sglang.srt.speculative.dflash_utils.top_p_renorm_prob", None
        ):
            result = build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=1,
                bs=2,
            )

        expected = torch.tensor(
            [
                [0.7310586, 0.2689414, 0.0, 0.0],
                [0.0, 0.0, 0.2689414, 0.7310586],
            ]
        )
        torch.testing.assert_close(result[:, 0], expected)
        torch.testing.assert_close(result.sum(dim=-1), torch.ones((2, 1)))

    def test_top_k_uses_torch_fallback_without_sgl_kernel(self):
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        sampling_info = SimpleNamespace(
            need_top_k_sampling=True,
            need_top_p_sampling=False,
            temperatures=torch.ones((1, 1)),
            top_ks=torch.tensor([2]),
        )

        with patch(
            "sglang.srt.speculative.dflash_utils.top_k_renorm_prob", None
        ):
            result = build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=1,
                bs=1,
                use_sparse_topk=False,
            )

        expected = torch.tensor([[[0.0, 0.7310586, 0.0, 0.2689414]]])
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.sum(dim=-1), torch.ones((1, 1)))


if __name__ == "__main__":
    unittest.main()
