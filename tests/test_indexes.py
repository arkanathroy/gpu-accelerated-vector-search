"""
tests/test_indexes.py
Sanity checks — run on CPU without GPU required.

  pytest tests/ -v
"""
from pathlib import Path
import numpy as np
import pytest

INDEX_DIR = Path("indexes")
EMB_DIR   = Path("embeddings")
K         = 10


@pytest.fixture(scope="module")
def query_embs():
    path = EMB_DIR / "query_embs.npy"
    if not path.exists():
        pytest.skip("query_embs.npy not found — run Step 4 first")
    return np.load(path).astype("float32")


@pytest.fixture(scope="module")
def hnsw_index():
    faiss = pytest.importorskip("faiss")
    path  = INDEX_DIR / "hnsw_cpu.faiss"
    if not path.exists():
        pytest.skip("hnsw_cpu.faiss not found — run Step 5 first")
    return faiss.read_index(str(path))


def test_hnsw_ntotal(hnsw_index):
    assert hnsw_index.ntotal == 200_000, \
        f"Expected 200000 vectors, got {hnsw_index.ntotal}"


def test_hnsw_search_shape(hnsw_index, query_embs):
    distances, indices = hnsw_index.search(query_embs[:10], K)
    assert distances.shape == (10, K)
    assert indices.shape == (10, K)


def test_hnsw_no_negative_ids(hnsw_index, query_embs):
    _, indices = hnsw_index.search(query_embs[:50], K)
    assert (indices >= 0).all(), "Found -1 (unfilled) IDs in HNSW search results"


def test_embeddings_normalised(query_embs):
    norms = np.linalg.norm(query_embs[:20], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_all_index_files_exist():
    expected = [
        "hnsw_cpu.faiss",
        "ivfpq_gpu.faiss",
        "cagra_gpu.faiss",
        "hnsw_from_cagra.faiss",
    ]
    missing = [f for f in expected if not (INDEX_DIR / f).exists()]
    if missing:
        pytest.skip(f"Index files not present (build on GPU first): {missing}")
    # assert not missing, f"Missing index files: {missing}"
