"""
03_build_indexes.py
────────────────────────────────────────────────────────────────
Builds three FAISS indexes and captures build time + energy.

Indexes produced:
  indexes/hnsw_cpu.faiss       CPU HNSW  (efConstruction=200, M=32)
  indexes/ivfpq_gpu.faiss      GPU IVF-PQ via cuVS  (nlist=512, M=32, nbits=8)
  indexes/cagra_gpu.faiss      GPU CAGRA via cuVS   (graph_degree=32)
  indexes/hnsw_from_cagra.faiss  CAGRA-built HNSW (bonus artifact)

Requires:
  conda install pytorch::faiss-gpu-cuvs  (T4 / CC ≥ 7.5)

Run:
  python src/03_build_indexes.py
  python src/03_build_indexes.py --gdrive
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pynvml

# ─── Energy Monitor ─────────────────────────────────────────────────────────

class GPUEnergyMonitor:
    """
    On Volta+ (T4 = Turing, which supports the energy API):
      Uses nvmlDeviceGetTotalEnergyConsumption — returns cumulative mJ since
      driver load. Delta between start/stop = energy for the interval.

    Requirement: pip install nvidia-ml-py
    """

    def __init__(self, device_index: int = 0):
        pynvml.nvmlInit()
        self.handle  = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.name    = pynvml.nvmlDeviceGetName(self.handle)
        self.cc      = pynvml.nvmlDeviceGetCudaComputeCapability(self.handle)
        self._start_mj: int = 0
        self._start_t:  float = 0.0
        print(f"[GPUEnergyMonitor] GPU : {self.name}  CC: {self.cc[0]}.{self.cc[1]}")

    def start(self):
        self._start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self._start_t  = time.perf_counter()

    def stop(self) -> dict:
        end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        end_t  = time.perf_counter()
        duration_s = end_t  - self._start_t
        energy_j   = (end_mj - self._start_mj) / 1_000.0   # mJ → J
        avg_watt   = energy_j / duration_s if duration_s > 0 else 0.0
        return {
            "duration_s": round(duration_s, 3),
            "energy_j":   round(energy_j,   2),
            "avg_watt":   round(avg_watt,    2),
        }

    def shutdown(self):
        pynvml.nvmlShutdown()


# ─── Index Builders ──────────────────────────────────────────────────────────

def build_hnsw_cpu(embeddings: np.ndarray, idx_dir: Path) -> dict:
    """FAISS HNSW on CPU — pure faiss-cpu, no GPU dependency."""
    out = idx_dir / "hnsw_cpu.faiss"
    if out.exists():
        print(f"  [SKIP] {out} exists.")
        return {}

    D = embeddings.shape[1]
    print("  Building CPU HNSW (M=32, efConstruction=200)...")
    t0    = time.perf_counter()
    index = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    elapsed = time.perf_counter() - t0
    index.hnsw.efSearch = 128   # default search quality; tunable
    faiss.write_index(index, str(out))
    print(f"  ✓ HNSW: {elapsed:.1f}s  ntotal={index.ntotal:,}")
    return {"build_s": round(elapsed, 2), "ntotal": int(index.ntotal)}


def build_ivfpq_gpu(embeddings: np.ndarray, idx_dir: Path,
                    res, monitor: GPUEnergyMonitor) -> dict:
    """
    GPU IVF-PQ using cuVS backend.

    nlist=512  : Voronoi cells ≈ sqrt(200K)
    M=32       : PQ sub-vectors (384 / 32 = 12 dims each)
    nbits=8    : 256 centroids per sub-vector
    → 32 bytes/vector (vs 1536 bytes raw) = 48× compression
    """
    out = idx_dir / "ivfpq_gpu.faiss"
    if out.exists():
        print(f"  [SKIP] {out} exists.")
        return {}

    D, N = embeddings.shape[1], len(embeddings)
    nlist, M, nbits = 512, 32, 8

    cfg = faiss.GpuIndexIVFPQConfig()
    cfg.use_cuvs = True

    print(f"  Building GPU IVF-PQ (nlist={nlist}, M={M}, nbits={nbits})...")
    monitor.start()
    gpu_index = faiss.GpuIndexIVFPQ(
        res, D, nlist, M, nbits,
        faiss.METRIC_INNER_PRODUCT, cfg
    )
    train_n = min(N, 40 * nlist)   # 40× nlist = 20,480 vectors
    print(f"  Training on {train_n:,} vectors...")
    gpu_index.train(embeddings[:train_n])
    gpu_index.add(embeddings)
    stats = monitor.stop()

    gpu_index.nprobe = 16   # search 16/512 cells; tune via nprobe sweep
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)   # must serialise as CPU
    faiss.write_index(cpu_index, str(out))
    print(f"  ✓ IVF-PQ: {stats['duration_s']:.1f}s  "
          f"energy={stats['energy_j']:.1f}J  ntotal={gpu_index.ntotal:,}")
    return {**stats, "ntotal": int(gpu_index.ntotal)}


def build_cagra_gpu(embeddings: np.ndarray, idx_dir: Path,
                    res, monitor: GPUEnergyMonitor) -> dict:
    """
    GPU CAGRA using cuVS backend.

    Algorithm:
      1. Build intermediate kNN graph (intermediate_graph_degree=64 neighbours)
      2. Prune to graph_degree=32 final edges per node
      3. Parallel beam search at query time across CUDA cores

    Key difference from IVF-PQ:
      - CAGRA uses train() to index ALL data; there is no add() step.
      - graph_build_algo_IVF_PQ is the scalable build algorithm for >100K vectors.

    VRAM peak during build: N × (D×2 + 276) bytes
      = 200,000 × (384×2 + 276) ≈ 200 MB  (well within 16 GB T4)
    """
    out = idx_dir / "cagra_gpu.faiss"
    if out.exists():
        print(f"  [SKIP] {out} exists.")
        return {}

    D = embeddings.shape[1]

    cfg = faiss.GpuIndexCagraConfig()
    cfg.graph_degree              = 32
    cfg.intermediate_graph_degree = 64
    cfg.build_algo = faiss.graph_build_algo_IVF_PQ

    print("  Building GPU CAGRA (graph_degree=32, intermediate=64)...")
    monitor.start()
    cagra = faiss.GpuIndexCagra(res, D, faiss.METRIC_INNER_PRODUCT, cfg)
    cagra.train(embeddings)   # CAGRA: train() indexes everything
    stats = monitor.stop()

    cpu_index = faiss.index_gpu_to_cpu(cagra)
    faiss.write_index(cpu_index, str(out))
    print(f"  ✓ CAGRA: {stats['duration_s']:.1f}s  "
          f"energy={stats['energy_j']:.1f}J  ntotal={cagra.ntotal:,}")

    # Bonus: export CAGRA graph as CPU HNSW
    # Build quality is equivalent to several minutes of CPU HNSW construction.
    hnsw_out = idx_dir / "hnsw_from_cagra.faiss"
    hnsw = faiss.IndexHNSWCagra()
    hnsw.base_level_only = False
    cagra.copyTo(hnsw)
    faiss.write_index(hnsw, str(hnsw_out))
    print(f"  ✓ CAGRA→HNSW export saved → {hnsw_out}")

    return {**stats, "ntotal": int(cagra.ntotal)}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive", action="store_true")
    args = parser.parse_args()

    base    = Path("/content/drive/MyDrive/gpu-vector-search") if args.gdrive else Path(".")
    emb_dir = base / "embeddings"
    idx_dir = base / "indexes"
    res_dir = base / "benchmarks"
    idx_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    embeddings = np.load(emb_dir / "passage_embs.npy").astype("float32")
    N, D = embeddings.shape
    print(f"Dataset: {N:,} vectors × dim={D}  ({N*D*4/1024**2:.0f} MB raw)\n")

    monitor = GPUEnergyMonitor(device_index=0)
    res     = faiss.StandardGpuResources()

    results = {}

    print("=" * 55)
    print("STEP 1 / 3  CPU HNSW (baseline)")
    print("=" * 55)
    results["hnsw_cpu"] = build_hnsw_cpu(embeddings, idx_dir)

    print("\n" + "=" * 55)
    print("STEP 2 / 3  GPU IVF-PQ (cuVS)")
    print("=" * 55)
    results["ivfpq_gpu"] = build_ivfpq_gpu(embeddings, idx_dir, res, monitor)

    print("\n" + "=" * 55)
    print("STEP 3 / 3  GPU CAGRA (cuVS)")
    print("=" * 55)
    results["cagra_gpu"] = build_cagra_gpu(embeddings, idx_dir, res, monitor)

    monitor.shutdown()

    out_json = res_dir / "build_stats.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Build stats → {out_json}")


if __name__ == "__main__":
    main()
