"""Unit tests for the scheduler's prompt-length check (validate_input_length).

Turns red if the rejection loses OpenAI's context-length sentence, which
OpenAI-compatible clients use to tell a context overflow from other 400s, or if
the auto-truncate and in-range paths change.
"""

import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.utils import (
    validate_capture_start_lens,
    validate_input_length,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(num_tokens):
    return SimpleNamespace(origin_input_ids=array("q", range(num_tokens)))


class TestValidateInputLength(CustomTestCase):
    def test_rejection_uses_openai_context_length_wording(self):
        req = _req(1500)
        msg = validate_input_length(
            req, max_req_input_len=1018, allow_auto_truncate=False
        )
        self.assertIn(
            "Input length (1500 tokens) exceeds the maximum allowed length (1017 tokens).",
            msg,
        )
        self.assertIn("This model's maximum context length is 1017 tokens", msg)
        self.assertIn("--allow-auto-truncate", msg)
        self.assertEqual(len(req.origin_input_ids), 1500)

    def test_limit_itself_is_rejected(self):
        msg = validate_input_length(
            _req(1018), max_req_input_len=1018, allow_auto_truncate=False
        )
        self.assertIn("This model's maximum context length is 1017 tokens", msg)

    def test_auto_truncate_unchanged(self):
        req = _req(1500)
        self.assertIsNone(
            validate_input_length(req, max_req_input_len=1018, allow_auto_truncate=True)
        )
        self.assertEqual(list(req.origin_input_ids), list(range(1018)))

    def test_in_range_unchanged(self):
        req = _req(1017)
        for allow_auto_truncate in (False, True):
            self.assertIsNone(
                validate_input_length(
                    req, max_req_input_len=1018, allow_auto_truncate=allow_auto_truncate
                )
            )
        self.assertEqual(len(req.origin_input_ids), 1017)


def _capture_req(*, routed=None, indexer=None):
    """A request asking for each capture whose start_len is not None."""
    return SimpleNamespace(
        return_routed_experts=routed is not None,
        routed_experts_start_len=routed or 0,
        return_indexer_topk=indexer is not None,
        indexer_topk_start_len=indexer or 0,
    )


class TestValidateCaptureStartLens(CustomTestCase):
    def test_start_lens_must_lie_in_zero_to_prompt_tokens(self):
        for name, key in (
            ("routed_experts_start_len", "routed"),
            ("indexer_topk_start_len", "indexer"),
        ):
            for start_len, error in (
                (0, None),
                (10, None),
                (-1, "is lower than 0"),
                (11, "is higher than the number of input tokens"),
            ):
                with self.subTest(name=name, start_len=start_len):
                    msg = validate_capture_start_lens(
                        recv_req=_capture_req(**{key: start_len}),
                        num_input_tokens=10,
                    )
                    if error is None:
                        self.assertIsNone(msg)
                    else:
                        self.assertIn(f"{name}={start_len} {error}", msg)

    def test_start_len_of_an_unrequested_capture_is_ignored(self):
        recv_req = _capture_req(routed=5)
        recv_req.indexer_topk_start_len = -3
        self.assertIsNone(
            validate_capture_start_lens(recv_req=recv_req, num_input_tokens=10)
        )


if __name__ == "__main__":
    unittest.main()
