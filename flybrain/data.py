"""Fetch and cache the public FlyWire adult Drosophila connectome (release 783)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CACHE = Path.home() / ".cache" / "flybrain"

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
# FlyWire 783 proofread connections, Zenodo record 10676866 (~150 MB)
CONNECTIONS_URL = (
    "https://zenodo.org/api/records/10676866/files/"
    "proofread_connections_783.feather/content"
)

# Sign of each neuron's output, following Shiu et al. 2024 (Nature).
EXCITATORY = {"acetylcholine", "dopamine", "serotonin", "octopamine"}
INHIBITORY = {"gaba", "glutamate"}


def _resolve_columns(con: pd.DataFrame) -> tuple[str, str, str]:
    """Work out which columns hold pre id, post id and synapse count.

    The FlyWire releases have shipped these under several names
    (pre_root_id / pre_pt_root_id / pre, syn_count / count / weight),
    so match on substrings rather than assuming one spelling.
    """
    cols = list(con.columns)
    low = {c: str(c).lower() for c in cols}

    def pick(*musts, avoid=()):
        for c in cols:
            n = low[c]
            if all(m in n for m in musts) and not any(a in n for a in avoid):
                return c
        return None

    pre = pick("pre", "root") or pick("pre")
    post = pick("post", "root") or pick("post")
    cnt = (pick("syn", "count") or pick("count") or pick("weight")
           or pick("syn"))
    if not (pre and post and cnt):
        raise KeyError(
            "could not identify the pre/post/count columns in the connection "
            f"table. Columns present: {cols}"
        )
    print(f"using columns: pre={pre!r} post={post!r} count={cnt!r}")
    return pre, post, cnt


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {dest.name} ...")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def download(cache: Path = CACHE) -> tuple[Path, Path]:
    ann = cache / "annotations.tsv"
    con = cache / "connections_783.feather"
    if not ann.exists():
        _download(ANNOTATIONS_URL, ann)
    if not con.exists():
        _download(CONNECTIONS_URL, con)
    return ann, con


def load(cache: Path = CACHE):
    """Return (annotations DataFrame, signed weight matrix as CSR, index map).

    The matrix is oriented so that row i holds the outgoing weights of neuron i:
    W[pre, post] = +/- synapse_count, sign taken from the presynaptic transmitter.
    """
    cache = Path(cache)
    npz = cache / "weights.npz"
    ann_path, con_path = download(cache)
    ann = pd.read_csv(ann_path, sep="\t", low_memory=False)
    ann = ann.drop_duplicates("root_id").reset_index(drop=True)
    index = pd.Series(ann.index.values, index=ann.root_id.values)

    if npz.exists():
        W = sp.load_npz(npz).tocsr()
        return ann, W, index

    con = pd.read_feather(con_path)
    pre_c, post_c, cnt_c = _resolve_columns(con)
    pre = con[pre_c].values
    post = con[post_c].values
    cnt = con[cnt_c].values.astype(np.float32)

    # map FlyWire root ids onto row numbers, dropping edges whose endpoints
    # are not in the annotation table
    pi = index.reindex(pre).values
    qi = index.reindex(post).values
    ok = ~(pd.isna(pi) | pd.isna(qi))
    dropped = int((~ok).sum())
    if dropped:
        print(f"dropped {dropped:,} edges with unannotated endpoints")
    pi, qi, cnt = pi[ok].astype(np.int32), qi[ok].astype(np.int32), cnt[ok]

    nt = ann.top_nt.fillna("acetylcholine").str.lower().values
    sign = np.where(np.isin(nt, list(INHIBITORY)), -1.0, 1.0).astype(np.float32)
    cnt = cnt * sign[pi]

    n = len(ann)
    W = sp.coo_matrix((cnt, (pi, qi)), shape=(n, n)).tocsr()
    W.sum_duplicates()
    npz.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(npz, W)
    return ann, W, index
