"""Unit tests for srt/state_capturer/indexer_diff.py: the indexer-diff-v1
codec, its serial and parallel encoders, and the response encode/decode helpers.
"""

import struct
import unittest
from unittest.mock import patch

import numba
import numpy as np
import pybase64
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.arg_groups.choices import INDEXER_TOPK_ENCODING_CHOICES
from sglang.srt.state_capturer import indexer_diff
from sglang.srt.state_capturer.indexer_diff import (
    decode,
    encode,
    encode_indexer_topk_b64,
    encode_indexer_topk_for_response,
    encode_parallel,
    encode_serial,
    indexer_topk_format,
    pack,
    sorted_rows,
)
from sglang.srt.state_capturer.indexer_topk import extract_indexer_topk_from_meta_info

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _walk_rows(*, seed, n, num_layers, topk, max_id, churn=0.1, pad=True):
    """Rows like a decode: each row keeps most of the previous row's ids per
    layer; early rows may hold fewer than topk ids (-1 padded)."""
    rng = np.random.default_rng(seed)
    rows = np.full((n, num_layers, topk), -1, np.int32)
    for layer in range(num_layers):
        cur = rng.choice(max_id + 1, size=topk, replace=False)
        for r in range(n):
            swap = rng.random(topk) < churn
            pool = np.setdiff1d(np.arange(max_id + 1), cur)
            cur = cur.copy()
            cur[swap] = rng.choice(pool, size=int(swap.sum()), replace=False)
            valid = rng.integers(1, topk + 1) if pad and r < 3 else topk
            rows[r, layer, :valid] = rng.permutation(cur)[:valid]
    return rows


def _blob(rows, *, compress=False, encoder=encode):
    flags, counts, ids = encoder(rows)
    return pack(
        num_layers=rows.shape[1],
        topk=rows.shape[2],
        flags=flags,
        counts=counts,
        ids=ids,
        compress=compress,
    )


class TestIndexerDiffCodec(CustomTestCase):
    def test_blob_bytes_match_the_vllm_format(self):
        """indexer-diff-v1 is shared with vLLM's encoder; these bytes were
        produced by vllm/distributed/aux_output_connector/indexer_diff.py."""
        rows = np.array(
            [
                [[3, 1, -1], [7, -1, -1]],
                [[1, 4, -1], [7, 2, -1]],
                [[4, 1, 3], [2, -1, -1]],
            ],
            np.int32,
        )
        golden = bytes.fromhex(
            "494458440100000003000000020003000800000000000000"
            "01000002000000010000000100010001000000010000000000"
            "010003000100070004000300020003000700"
        )
        for encoder in (encode_serial, encode_parallel):
            with self.subTest(encoder=encoder.__name__):
                self.assertEqual(_blob(rows, encoder=encoder), golden)
        self.assertEqual(
            decode(golden).tolist(),
            [
                [[1, 3, -1], [7, -1, -1]],
                [[1, 4, -1], [2, 7, -1]],
                [[1, 3, 4], [2, -1, -1]],
            ],
        )

    def test_round_trip_is_lossless_as_sets(self):
        rows = _walk_rows(seed=0, n=300, num_layers=4, topk=64, max_id=2000)
        for compress in (False, True):
            with self.subTest(compress=compress):
                blob = _blob(rows, compress=compress)
                np.testing.assert_array_equal(decode(blob), sorted_rows(rows))

    def test_parallel_encoder_is_byte_identical_to_serial(self):
        """Holds whether or not rows are split across tasks (fewer layers than
        threads), and for duplicate ids, which both encoders must repeat."""
        rng = np.random.default_rng(1)
        walk = _walk_rows(seed=2, n=97, num_layers=3, topk=32, max_id=65535)
        dups = rng.integers(-1, 12, size=(41, 5, 16)).astype(np.int32)
        threads = numba.get_num_threads()
        for name, rows in (("walk", walk), ("duplicates", dups)):
            expected = _blob(rows, encoder=encode_serial)
            for num_threads in sorted({1, 3, threads}):
                with self.subTest(rows=name, threads=num_threads):
                    numba.set_num_threads(num_threads)
                    try:
                        got = _blob(rows, encoder=encode_parallel)
                    finally:
                        numba.set_num_threads(threads)
                    self.assertEqual(got, expected)

    def test_serial_retry_after_full_ids_buffer_starts_clean(self):
        """When the first ids buffer fills mid-row, the retry must not see
        the aborted attempt's marks; the blob must not depend on the guess."""
        rows = _walk_rows(seed=3, n=20, num_layers=2, topk=16, max_id=100, pad=False)
        expected = _blob(rows, encoder=encode_serial)
        n, num_layers, topk = rows.shape
        # Row 0 fills the first buffer; row 1 aborts after its first added id.
        tight = (num_layers * topk + 1, n * num_layers * topk * 2)
        with patch.object(indexer_diff, "_ids_capacities", return_value=tight):
            self.assertEqual(_blob(rows, encoder=encode_serial), expected)

    def test_ids_above_uint16_switch_the_blob_to_uint32_ids(self):
        """DeepSeek-V3.2 picks are token positions, so past 64K tokens an id
        exceeds uint16; that blob alone must store uint32 ids and still decode,
        and a decoder must reject header bits it does not know."""
        narrow = np.array([[[0, 1]], [[0, 65535]]], np.int32)
        wide = narrow.copy()
        wide[1, 0, 1] = 65536
        for encoder in (encode_serial, encode_parallel):
            for rows, wide_bit in ((narrow, 0), (wide, 2)):
                with self.subTest(encoder=encoder.__name__, wide_bit=wide_bit):
                    blob = _blob(rows, encoder=encoder)
                    self.assertEqual(struct.unpack_from("<4sHH", blob)[2], wide_bit)
                    np.testing.assert_array_equal(decode(blob), sorted_rows(rows))
        unknown_bit = bytearray(_blob(narrow, encoder=encode_serial))
        unknown_bit[6] |= 4
        with self.assertRaises(ValueError):
            decode(bytes(unknown_bit))

    def test_malformed_blob_raises_instead_of_writing_out_of_bounds(self):
        """A row-layer whose kept plus added ids exceed topk (as from rows with
        duplicate ids) must be rejected; the decode kernel does no bounds checks."""
        rows = np.array([[[0, 0]], [[0, 9]]], np.int32)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            decode(_blob(rows))

    def test_empty_request_round_trips(self):
        """start_len == seqlen - 1 leaves no rows; the shape must survive."""
        rows = np.empty((0, 3, 8), np.int32)
        for encoder in (encode_serial, encode_parallel):
            for compress in (False, True):
                with self.subTest(encoder=encoder.__name__, compress=compress):
                    out = decode(_blob(rows, compress=compress, encoder=encoder))
                    self.assertEqual(out.shape, (0, 3, 8))
                    self.assertEqual(out.dtype, np.int32)


class TestIndexerTopkResponseEncoding(CustomTestCase):
    def test_every_server_encoding_has_a_format(self):
        """The format names are the trainer-facing contract shared with vLLM."""
        self.assertEqual(
            {e: indexer_topk_format(e) for e in INDEXER_TOPK_ENCODING_CHOICES},
            {
                "raw": "indexer-int32-v1",
                "diff": "indexer-diff-v1",
                "diff+zlib": "indexer-diff-v1+zlib",
            },
        )

    def test_client_helper_decodes_every_encoding(self):
        rows = _walk_rows(seed=4, n=50, num_layers=3, topk=16, max_id=500)
        for encoding in INDEXER_TOPK_ENCODING_CHOICES:
            with self.subTest(encoding=encoding):
                meta_info = {
                    "indexer_topk": encode_indexer_topk_b64(
                        rows=rows, encoding=encoding
                    ),
                    "indexer_topk_format": indexer_topk_format(encoding),
                }
                out = extract_indexer_topk_from_meta_info({"meta_info": meta_info})
                if encoding == "raw":
                    np.testing.assert_array_equal(out.reshape(rows.shape), rows)
                else:
                    np.testing.assert_array_equal(out, sorted_rows(rows))

    def test_raw_bytes_and_missing_format_keep_the_legacy_contract(self):
        """Raw is the pre-diff wire format; a response from an older server
        (no indexer_topk_format) must still decode as raw int32."""
        rows = torch.randint(-1, 1000, (7, 3, 5), dtype=torch.int32)
        legacy = pybase64.b64encode(rows.numpy().tobytes()).decode("utf-8")
        self.assertEqual(
            encode_indexer_topk_b64(rows=rows.numpy(), encoding="raw"), legacy
        )
        out = extract_indexer_topk_from_meta_info(
            {"meta_info": {"indexer_topk": legacy}}
        )
        np.testing.assert_array_equal(out.reshape(7, 3, 5), rows.numpy())

    def test_unencodable_rows_drop_the_request_instead_of_raising(self):
        rows = np.full((2, 2), 7, np.int32)  # not [n, L, K]
        with self.assertLogs(indexer_diff.logger, level="ERROR"):
            self.assertIsNone(
                encode_indexer_topk_for_response(rows=rows, encoding="diff", rid="r")
            )
        self.assertIsNotNone(
            encode_indexer_topk_for_response(rows=rows, encoding="raw", rid="r")
        )


if __name__ == "__main__":
    unittest.main()
