"""A fake connectome with the real annotations but randomly drawn wiring.

Only for testing the engine end to end when you have not downloaded the real
connection table yet. Results from it mean nothing biologically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import CACHE, INHIBITORY, download


def make(mean_out_degree: int = 20, seed: int = 0):
    ann_path, _ = _annotations_only()
    ann = pd.read_csv(ann_path, sep="\t", low_memory=False)
    ann = ann.drop_duplicates("root_id").reset_index(drop=True)
    n = len(ann)
    rng = np.random.default_rng(seed)
    nnz = n * mean_out_degree
    pre = rng.integers(0, n, nnz, dtype=np.int32)
    post = rng.integers(0, n, nnz, dtype=np.int32)
    cnt = rng.integers(1, 12, nnz).astype(np.float32)
    nt = ann.top_nt.fillna("acetylcholine").str.lower().values
    sign = np.where(np.isin(nt, list(INHIBITORY)), -1.0, 1.0).astype(np.float32)
    W = sp.coo_matrix((cnt * sign[pre], (pre, post)), shape=(n, n)).tocsr()
    W.sum_duplicates()
    index = pd.Series(ann.index.values, index=ann.root_id.values)
    return ann, W, index


def _annotations_only():
    from .data import ANNOTATIONS_URL, _download
    ann = CACHE / "annotations.tsv"
    if not ann.exists():
        _download(ANNOTATIONS_URL, ann)
    return ann, None
