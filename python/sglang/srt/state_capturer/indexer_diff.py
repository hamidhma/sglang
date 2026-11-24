"""Diff encoding for indexer top-k rows (format ``indexer-diff-v1``).

Each row of an ``indexer-int32-v1`` array [n, L, K] holds, per indexer layer,
the set of selected ids (``-1`` padded). Consecutive rows share most ids, so a
row is stored as the ids it adds and drops relative to the previous row.

Blob layout (one ``uint8`` buffer, little endian)::

    header  <4sHHIHHQ: magic b"IDXD", version, bits (1 = zlib body,
            2 = uint32 ids), n_rows, num_layers, topk, n_ids
    body    flags[n_rows] uint8            1 = keyframe (diff vs empty set)
            counts[n_rows, L, 2] uint16    (n_added, n_dropped) per layer
            ids[n_ids] uint16 or uint32    per row, per layer: added, dropped

The layout is shared with vLLM's ``aux_output_connector/indexer_diff.py`` so
one trainer-side decoder reads both engines; keep the two byte-identical.
Decoding yields the same sets, each row sorted ascending and ``-1`` padded.
Ids are uint16 unless a request holds an id above 65535 (e.g. DeepSeek-V3.2
token positions past 64K); then that blob alone stores uint32 ids. Only
numpy, numba and pybase64 are required.
"""

from __future__ import annotations

import logging
import struct
import zlib
from typing import Optional

import numba
import numpy as np
import pybase64
from numba import njit, prange

logger = logging.getLogger(__name__)

RAW_FORMAT = "indexer-int32-v1"
DIFF_FORMAT = "indexer-diff-v1"
DIFF_ZLIB_FORMAT = DIFF_FORMAT + "+zlib"

_FORMAT_BY_ENCODING = {
    "raw": RAW_FORMAT,
    "diff": DIFF_FORMAT,
    "diff+zlib": DIFF_ZLIB_FORMAT,
}

_MAGIC = b"IDXD"
_VERSION = 1
_HEADER = struct.Struct("<4sHHIHHQ")
_ZLIB_BIT = 1
_WIDE_IDS_BIT = 2
_KNOWN_BITS = _ZLIB_BIT | _WIDE_IDS_BIT
_ZLIB_LEVEL = 6
_MAX_NARROW_ID = np.iinfo(np.uint16).max
# Below this many (row, layer) pairs the thread fan-out costs more than it
# saves; measured on a 20-core GB10 (crossover at 2-4 rows of 21 layers).
_PARALLEL_MIN_ROW_LAYERS = 64


@njit(nogil=True, cache=True)
def _encode_serial_kernel(rows, mark, flags, counts, ids):
    """Fill flags/counts/ids; return the number of ids, or -1 if ids is full.

    Same algorithm as vLLM's ``_encode_kernel`` with no previous row.
    """
    n, num_layers, topk = rows.shape
    pos = 0
    cap = ids.shape[0]
    for r in range(n):
        keyframe = r == 0
        flags[r] = 1 if keyframe else 0
        for layer in range(num_layers):
            cur = rows[r, layer]
            # Added: in cur, not in base.
            if not keyframe:
                base = rows[r - 1, layer]
                for j in range(topk):
                    v = base[j]
                    if v >= 0:
                        mark[v] = 1
            n_add = 0
            for j in range(topk):
                v = cur[j]
                if v >= 0 and mark[v] == 0:
                    if pos >= cap:
                        return -1
                    ids[pos] = v
                    pos += 1
                    n_add += 1
            n_del = 0
            if not keyframe:
                for j in range(topk):
                    v = base[j]
                    if v >= 0:
                        mark[v] = 0
                # Dropped: in base, not in cur.
                for j in range(topk):
                    v = cur[j]
                    if v >= 0:
                        mark[v] = 1
                for j in range(topk):
                    v = base[j]
                    if v >= 0 and mark[v] == 0:
                        if pos >= cap:
                            return -1
                        ids[pos] = v
                        pos += 1
                        n_del += 1
                for j in range(topk):
                    v = cur[j]
                    if v >= 0:
                        mark[v] = 0
            counts[r, layer, 0] = n_add
            counts[r, layer, 1] = n_del
    return pos


@njit(nogil=True, cache=True, inline="always")
def _stamp_row(rows, r, layer, stamps):
    row_stamps = stamps[r & 1]
    for v in rows[r, layer]:
        if v >= 0:
            row_stamps[v] = r


@njit(nogil=True, cache=True, inline="always")
def _diff_row_layer(rows, r, layer, stamps, out, pos):
    """Diff one (row, layer) against the row before it; return (n_add, n_del).

    Writes the added then the dropped ids to ``out[pos:]`` unless ``pos`` < 0.
    ``stamps[r & 1, v] == r`` iff v is in row r, so row r-1 must be stamped
    first; membership is set-based, so duplicates repeat as in the serial kernel.
    """
    _stamp_row(rows, r, layer, stamps)
    cur = rows[r, layer]
    n_add = 0
    if r == 0:
        for v in cur:
            if v >= 0:
                if pos >= 0:
                    out[pos + n_add] = v
                n_add += 1
        return n_add, 0
    base_stamps = stamps[(r - 1) & 1]
    for v in cur:
        if v >= 0 and base_stamps[v] != r - 1:
            if pos >= 0:
                out[pos + n_add] = v
            n_add += 1
    cur_stamps = stamps[r & 1]
    n_del = 0
    for v in rows[r - 1, layer]:
        if v >= 0 and cur_stamps[v] != r:
            if pos >= 0:
                out[pos + n_add + n_del] = v
            n_del += 1
    return n_add, n_del


@njit(nogil=True, cache=True, inline="always")
def _init_task_stamps(rows, start, layer, mark_size):
    stamps = np.full((2, mark_size), -1, np.int32)
    if start > 0:
        _stamp_row(rows, start - 1, layer, stamps)
    return stamps


@njit(nogil=True, cache=True, parallel=True)
def _max_id_kernel(rows):
    n = rows.shape[0]
    row_max = np.full(max(n, 1), -1, np.int64)
    for r in prange(n):
        row_max[r] = rows[r].max()
    return row_max.max()


@njit(nogil=True, cache=True, parallel=True)
def _count_parallel_kernel(rows, mark_size, chunk_rows, counts):
    n, num_layers, _ = rows.shape
    num_chunks = (n + chunk_rows - 1) // chunk_rows
    for task in prange(num_layers * num_chunks):
        layer = task % num_layers
        start = (task // num_layers) * chunk_rows
        stamps = _init_task_stamps(rows, start, layer, mark_size)
        for r in range(start, min(n, start + chunk_rows)):
            n_add, n_del = _diff_row_layer(rows, r, layer, stamps, counts[0, 0], -1)
            counts[r, layer, 0] = n_add
            counts[r, layer, 1] = n_del


@njit(nogil=True, cache=True, parallel=True)
def _write_parallel_kernel(rows, mark_size, chunk_rows, offsets, ids):
    n, num_layers, _ = rows.shape
    num_chunks = (n + chunk_rows - 1) // chunk_rows
    for task in prange(num_layers * num_chunks):
        layer = task % num_layers
        start = (task // num_layers) * chunk_rows
        stamps = _init_task_stamps(rows, start, layer, mark_size)
        for r in range(start, min(n, start + chunk_rows)):
            _diff_row_layer(rows, r, layer, stamps, ids, offsets[r, layer])


@njit(nogil=True, cache=True)
def _decode_kernel(flags, counts, ids, topk, max_id, out):
    n, num_layers, _ = counts.shape
    state = np.full((num_layers, topk), -1, np.int64)
    size = np.zeros(num_layers, np.int64)
    dropped = np.zeros(max_id + 1, np.uint8)
    kept = np.empty(topk, np.int64)
    pos = 0
    for r in range(n):
        if flags[r]:
            size[:] = 0
        for layer in range(num_layers):
            n_add = counts[r, layer, 0]
            n_del = counts[r, layer, 1]
            # numba does not bounds-check; a malformed blob must not read past
            # the ids or write past the topk slots of a row.
            if pos + n_add + n_del > ids.shape[0]:
                return -1
            added = np.sort(ids[pos : pos + n_add].astype(np.int64))
            pos += n_add
            for j in range(n_del):
                dropped[ids[pos + j]] = 1
            n_kept = 0
            for j in range(size[layer]):
                v = state[layer, j]
                if dropped[v] == 0:
                    kept[n_kept] = v
                    n_kept += 1
            for j in range(n_del):
                dropped[ids[pos + j]] = 0
            pos += n_del
            if n_kept + n_add > topk:
                return -1
            # Merge the sorted kept ids with the sorted added ids.
            i = 0
            k = 0
            m = 0
            while i < n_kept or k < n_add:
                if k >= n_add or (i < n_kept and kept[i] < added[k]):
                    state[layer, m] = kept[i]
                    i += 1
                else:
                    state[layer, m] = added[k]
                    k += 1
                m += 1
            size[layer] = m
            for j in range(m):
                out[r, layer, j] = state[layer, j]
    return pos


def _prepare_rows(rows: np.ndarray, *, parallel: bool) -> tuple[np.ndarray, int]:
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    if rows.ndim != 3:
        raise ValueError(f"indexer rows must be [n, layers, topk], got {rows.shape}")
    if parallel and rows.size:
        max_id = int(_max_id_kernel(rows))
    else:
        max_id = int(rows.max(initial=-1))
    return rows, max_id


def _ids_dtype(max_id: int) -> type:
    return np.uint16 if max_id <= _MAX_NARROW_ID else np.uint32


def _ids_capacities(n: int, num_layers: int, topk: int) -> tuple[int, int]:
    # Neighbouring rows usually differ by a small fraction; the second
    # capacity covers any input (each row adds and drops at most topk ids).
    return (num_layers * topk + n * num_layers * topk // 4, n * num_layers * topk * 2)


def encode_serial(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-threaded :func:`encode`; the byte-for-byte reference."""
    rows, max_id = _prepare_rows(rows, parallel=False)
    n, num_layers, topk = rows.shape
    mark = np.zeros(max_id + 1, np.uint8)
    flags = np.empty(n, np.uint8)
    counts = np.empty((n, num_layers, 2), np.uint16)
    for cap in _ids_capacities(n, num_layers, topk):
        ids = np.empty(cap, _ids_dtype(max_id))
        mark[:] = 0  # an aborted attempt leaves marks behind
        used = _encode_serial_kernel(rows, mark, flags, counts, ids)
        if used >= 0:
            return flags, counts, ids[:used].copy()
    raise AssertionError("unreachable: ids capacity covers every row")


def _chunk_rows(n: int, num_layers: int) -> int:
    # A row only diffs against the row before it, so (layer, row chunk) tasks
    # are independent. Rows are split only when layers alone leave threads
    # idle; finer splits measured slower on GB10 for 21, 40 and 61 layers.
    num_chunks = -(-numba.get_num_threads() // num_layers)
    return max(1, -(-n // num_chunks))


def encode_parallel(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass :func:`encode`, parallel over (layer, row chunk).

    Pass 1 counts the ids of every (row, layer), a prefix sum turns the counts
    into write offsets, and pass 2 writes the ids; the output equals
    :func:`encode_serial` byte for byte.
    """
    rows, max_id = _prepare_rows(rows, parallel=True)
    n, num_layers, _ = rows.shape
    flags = np.zeros(n, np.uint8)
    flags[:1] = 1
    counts = np.zeros((n, num_layers, 2), np.uint16)
    if n == 0 or num_layers == 0:
        return flags, counts, np.empty(0, np.uint16)
    chunk_rows = _chunk_rows(n, num_layers)
    _count_parallel_kernel(rows, max_id + 1, chunk_rows, counts)
    per_row_layer = counts.sum(axis=2, dtype=np.int64).ravel()
    ends = np.cumsum(per_row_layer)
    offsets = (ends - per_row_layer).reshape(n, num_layers)
    ids = np.empty(int(ends[-1]), _ids_dtype(max_id))
    _write_parallel_kernel(rows, max_id + 1, chunk_rows, offsets, ids)
    return flags, counts, ids


def encode(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diff-encode ``rows`` [n, L, K] into (flags, counts, ids); row 0 is a keyframe."""
    if rows.ndim == 3 and rows.shape[0] * rows.shape[1] < _PARALLEL_MIN_ROW_LAYERS:
        return encode_serial(rows)
    return encode_parallel(rows)


def pack(
    *,
    num_layers: int,
    topk: int,
    flags: np.ndarray,
    counts: np.ndarray,
    ids: np.ndarray,
    compress: bool = False,
) -> bytes:
    """Join one encoded request into a self-describing blob."""
    wide = ids.dtype.itemsize > 2
    body = (
        flags.astype(np.uint8).tobytes()
        + counts.astype("<u2").tobytes()
        + ids.astype("<u4" if wide else "<u2").tobytes()
    )
    bits = _WIDE_IDS_BIT if wide else 0
    if compress:
        body = zlib.compress(body, _ZLIB_LEVEL)
        bits |= _ZLIB_BIT
    header = _HEADER.pack(
        _MAGIC, _VERSION, bits, len(flags), num_layers, topk, len(ids)
    )
    return header + body


def unpack(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (flags, counts, ids, topk) from a blob made by :func:`pack`."""
    data = bytes(blob)
    magic, version, bits, n, num_layers, topk, n_ids = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION or bits & ~_KNOWN_BITS:
        raise ValueError(f"not an {DIFF_FORMAT} blob")
    body = data[_HEADER.size :]
    if bits & _ZLIB_BIT:
        body = zlib.decompress(body)
    flags = np.frombuffer(body, np.uint8, n)
    off = n
    counts = np.frombuffer(body, "<u2", n * num_layers * 2, off).reshape(
        n, num_layers, 2
    )
    off += counts.nbytes
    ids = np.frombuffer(body, "<u4" if bits & _WIDE_IDS_BIT else "<u2", n_ids, off)
    return flags, counts, ids, topk


def decode(blob: bytes) -> np.ndarray:
    """Rebuild the [n, L, K] int32 rows (sorted ascending, -1 padded)."""
    flags, counts, ids, topk = unpack(blob)
    n, num_layers, _ = counts.shape
    out = np.full((n, num_layers, topk), -1, np.int32)
    if n == 0:
        return out
    if not flags[0]:
        raise ValueError("indexer diff blob must start with a keyframe")
    max_id = int(ids.max(initial=0))
    used = _decode_kernel(flags, counts, ids, topk, max_id, out)
    if used != len(ids):
        raise ValueError("indexer diff blob is inconsistent")
    return out


def sorted_rows(rows: np.ndarray) -> np.ndarray:
    """Canonical form of raw rows: each row's ids sorted ascending, -1 padded."""
    big = np.iinfo(np.int32).max
    out = np.sort(np.where(rows >= 0, rows, big), axis=-1)
    return np.where(out == big, -1, out).astype(np.int32)


def indexer_topk_format(encoding: str) -> str:
    """The ``meta_info["indexer_topk_format"]`` value of a server encoding."""
    return _FORMAT_BY_ENCODING[encoding]


def encode_indexer_topk_b64(*, rows: np.ndarray, encoding: str) -> str:
    """Base64 of one request's [n, L, K] int32 rows in the server encoding.

    ``raw`` is the int32 bytes as-is; the diff encodings are a :func:`pack` blob.
    """
    if encoding == "raw":
        data = rows.tobytes()
    else:
        _, num_layers, topk = rows.shape
        flags, counts, ids = encode(rows)
        data = pack(
            num_layers=num_layers,
            topk=topk,
            flags=flags,
            counts=counts,
            ids=ids,
            compress=encoding == "diff+zlib",
        )
    return pybase64.b64encode(data).decode("utf-8")


def decode_indexer_topk_b64(*, data: str, format: Optional[str]) -> np.ndarray:
    """Inverse of :func:`encode_indexer_topk_b64`; ``None`` format means raw.

    Raw returns the flat int32 ids in capture order (the caller reshapes);
    the diff formats return [n, L, K] with each row sorted and -1 padded.
    """
    blob = pybase64.b64decode(data.encode("utf-8"))
    if format is None or format == RAW_FORMAT:
        return np.frombuffer(blob, dtype=np.int32)
    if format not in (DIFF_FORMAT, DIFF_ZLIB_FORMAT):
        raise ValueError(f"unknown indexer_topk_format {format!r}")
    return decode(blob)


def encode_indexer_topk_for_response(
    *, rows: np.ndarray, encoding: str, rid: str
) -> Optional[str]:
    """:func:`encode_indexer_topk_b64`, or ``None`` if ``rows`` cannot be encoded.

    Malformed rows (not [n, L, K]) give that request no ``indexer_topk``
    rather than taking down the process that encodes it.
    """
    try:
        return encode_indexer_topk_b64(rows=rows, encoding=encoding)
    except ValueError as e:
        logger.error("Dropping indexer_topk of request %s: %s", rid, e)
        return None


def warm_up_indexer_topk_encoding(encoding: str) -> None:
    """Compile the encode kernels now so the first request does not pay for it."""
    if encoding == "raw":
        return
    rows = np.zeros((2, 1, 1), np.int32)
    encode_serial(rows)
    encode_parallel(rows)
