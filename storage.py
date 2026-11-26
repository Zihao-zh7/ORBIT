"""Read-only cache sizing and decimal-byte storage-budget checks."""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np


def _cache_request(data, splits, feature_dtype, max_origins):
    if isinstance(splits, str):
        raise ValueError('Cache splits must be a nonempty sequence of distinct split names')
    try:
        requested = tuple(splits)
        valid = (bool(requested) and all(isinstance(s, str) for s in requested)
                 and len(set(requested)) == len(requested)
                 and set(requested) <= {'train', 'val', 'test'})
    except TypeError:
        valid = False
    if not valid:
        raise ValueError('Cache splits must be a nonempty sequence of distinct train/val/test names')
    try:
        dtype = np.dtype(feature_dtype)
    except (TypeError, ValueError) as error:
        raise ValueError('Feature cache dtype must be float32 or float64') from error
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Feature cache dtype must be float32 or float64')
    if isinstance(max_origins, dict):
        if set(max_origins) - {'train', 'val', 'test'}:
            raise ValueError('Origin limits contain an unknown split')
        limits = max_origins
    else:
        limits = dict.fromkeys(requested, max_origins)
    if any(value is not None and (isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer)) or value < 1)
           for value in limits.values()):
        raise ValueError('Origin limits must be positive integers')
    origins = {}
    for split in requested:
        rows = data.origins(split)
        maximum = limits.get(split)
        origins[split] = rows if maximum is None else rows[:maximum]
        if not len(origins[split]):
            raise ValueError(f'{split} has no complete forecast origins')
    return requested, dtype, origins


def _cache_layouts(count, nodes, horizon, feature_dim, feature_dtype):
    shape = (count, nodes, horizon)
    return {'origins': ((count,), np.int64),
            'predictions': (shape, np.float32), 'targets': (shape, np.float32),
            'valid': (shape, np.bool_),
            'features': ((count, nodes, feature_dim), feature_dtype)}


def estimate_cache_bytes(data, feature_dim, *, splits=('train', 'val', 'test'),
                         feature_dtype='float64', max_origins=None):
    """Exact .npy file bytes, without allocating feature/prediction arrays.

    Includes array headers. Metadata, checkpoints, logs and other artifacts need
    their own reserve in the caller's preflight. No files are read or written.
    """
    if (isinstance(feature_dim, (bool, np.bool_))
            or not isinstance(feature_dim, (int, np.integer)) or feature_dim < 1):
        raise ValueError('Feature dimension must be a positive integer')
    requested, dtype, origins = _cache_request(data, splits, feature_dtype, max_origins)
    total = 0
    for split in requested:
        layouts = _cache_layouts(len(origins[split]), data.num_nodes, data.output_steps,
                                 int(feature_dim), dtype)
        for shape, array_dtype in layouts.values():
            array_dtype = np.dtype(array_dtype)
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, {
                'descr': np.lib.format.dtype_to_descr(array_dtype),
                'fortran_order': False, 'shape': shape,
            })
            total += len(header.getvalue()) + math.prod(shape) * array_dtype.itemsize
    return total


def ensure_storage_budget(root, additional_bytes, *, budget_bytes=10_000_000_000):
    """Reject a planned write exceeding the entire run root's byte budget.

    This is a read-only preflight, not a reservation against concurrent writers.
    Callers must serialize writes and include future non-cache files in their
    additional-byte reserve. A missing root is treated as empty, not created.
    """
    for name, value, minimum in [('additional_bytes', additional_bytes, 0),
                                  ('budget_bytes', budget_bytes, 1)]:
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')
    root = Path(root)
    if root.exists() and not root.is_dir():
        raise ValueError('Storage-budget root must be a directory')
    current = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    projected = current + int(additional_bytes)
    if projected > budget_bytes:
        raise ValueError(f'Storage budget exceeded: {current} existing + {additional_bytes} planned '
                         f'= {projected} bytes > {budget_bytes} byte budget')
    return {'root': str(root.resolve()), 'current_bytes': current,
            'additional_bytes': int(additional_bytes), 'projected_bytes': projected,
            'budget_bytes': int(budget_bytes)}
