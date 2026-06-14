"""
04_benchmark.py
────────────────────────────────────────────────────────────────
Measures for each index:
  - Recall@10 vs brute-force ground truth
  - Single-query P50 / P95 / P99 latency (ms)
  - Batch QPS at multiple batch sizes
  - Energy consumed per 1000 queries (Joules) via NVML
  - nprobe sweep for IVF-PQ (recall vs latency Pareto curve)

Saves:
  benchmarks/results.json

Run:
  python src/04_benchmark.py
  python src/04_benchmark.py --gdrive
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pynvml
import torch


# ─── GPU Energy (same helper as build script) ────────────────────────────────

class GPUEnergyMonitor:
    def __init__(self, device_index: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.name   = pynvml.nvmlDeviceGetName(self.handle)

    def _energy_mj(self) -> int:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def measure(self, fn, *args, **kwargs):
        """
        Call fn(*args, **kwargs), return (result, energy_j, duration_s).
        torch.cuda.synchronize() ensures GPU work is complete before
        reading the energy counter — critical for accuracy.
        """
        torch.cuda.synchronize()
        e0 = self._energy_mj()
        t0 = time.perf_counter()

        result = fn(*args, **kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        energy_j = (self._energy_mj() - e0) / 1_000.0
        return result, energy_j, elapsed

    def shutdown(self):
        pynvml.nvmlShutdown()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """
    pred_ids : (N, k) array of retrieved IDs
    gt_ids   : (N, k) ground-truth IDs from brute-force
    Returns   : recall@k as float in [0, 1]
    """
    hits = sum(
        len(set(p.tolist()) & set(g.tolist()))
        for p, g in zip(pred_ids, gt_ids)
    )
    return hits / (len(gt_ids) * k)


def latency_profile(index, queries: np.ndarray, k: int,
                    warmup: int = 50, trials: int = 500) -> dict:
    """Single-query latency statistics in milliseconds."""
    lats = []
    for i in range(warmup + trials):
        q = queries[i % len(queries)].reshape(1, -1)
        if hasattr(index, "getDevice"):   # GPU index
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        index.search(q, k)
        if hasattr(index, "getDevice"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            lats.append(ms)
    a = np.array(lats)
    return {
        "p50_ms":  round(float(np.percentile(a, 50)), 3),
        "p95_ms":  round(float(np.percentile(a, 95)), 3),
        "p99_ms":  round(float(np.percentile(a, 99)), 3),
        "mean_ms": round(float(a.mean()),              3),
    }


def batch_qps(index, queries: np.ndarray, k: int, batch_sizes: list) -> dict:
    out = {}
    for bs in batch_sizes:
        batch = queries[:bs]
        for _ in range(5):          # warmup
            index.search(batch, k)
        if hasattr(index, "getDevice"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            index.search(batch, k)
        if hasattr(index, "getDevice"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        out[str(bs)] = round((20 * bs) / elapsed, 1)
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive", action="store_true")
    args = parser.parse_args()

    base    = Path("/content/drive/MyDrive/gpu-vector-search") if args.gdrive else Path(".")
    emb_dir = base / "embeddings"
    idx_dir = base / "indexes"
    res_dir = base / "benchmarks"
    res_dir.mkdir(parents=True, exist_ok=True)

    K = 10
    BATCH_SIZES = [1, 10, 50, 100, 500, 1000]
    NPROBE_VALS = [1, 4, 8, 16, 32, 64, 128, 256]

    print("Loading embeddings...")
    passage_embs = np.load(emb_dir / "passage_embs.npy").astype("float32")
    query_embs   = np.load(emb_dir / "query_embs.npy").astype("float32")
    N, D = passage_embs.shape
    print(f"Corpus: {N:,} × {D}d   Queries: {len(query_embs)}")

    # ── GPU setup ──────────────────────────────────────────────────────
    res = faiss.StandardGpuResources()
    co  = faiss.GpuClonerOptions()
    co.use_cuvs = True

    monitor = GPUEnergyMonitor(device_index=0)
    print(f"GPU: {monitor.name}\n")

    # ── Ground truth (exact brute force on GPU) ────────────────────────
    print("Computing brute-force ground truth on GPU...")
    flat_gpu = faiss.GpuIndexFlatIP(res, D)
    flat_gpu.add(passage_embs)
    torch.cuda.synchronize()
    _, gt = flat_gpu.search(query_embs, K)
    print(f"  Ground truth shape: {gt.shape}\n")

    # ── Load indexes ───────────────────────────────────────────────────
    print("Loading indexes...")

    hnsw = faiss.read_index(str(idx_dir / "hnsw_cpu.faiss"))
    hnsw.hnsw.efSearch = 128

    ivfpq_cpu  = faiss.read_index(str(idx_dir / "ivfpq_gpu.faiss"))
    ivfpq_gpu  = faiss.index_cpu_to_gpu(res, 0, ivfpq_cpu, co)
    ivfpq_gpu.nprobe = 16

    cagra_cpu  = faiss.read_index(str(idx_dir / "cagra_gpu.faiss"))
    cagra_gpu  = faiss.index_cpu_to_gpu(res, 0, cagra_cpu, co)

    index_map = {
        "CPU HNSW":          hnsw,
        "GPU IVF-PQ (cuVS)": ivfpq_gpu,
        "GPU CAGRA (cuVS)":  cagra_gpu,
    }

    # ── Per-index benchmarks ───────────────────────────────────────────
    results = {}
    for name, idx in index_map.items():
        print(f"\n{'='*55}")
        print(f"Benchmarking: {name}")
        print(f"{'='*55}")

        # Recall@10
        _, pred = idx.search(query_embs, K)
        r10 = recall_at_k(pred, gt, K)
        print(f"  Recall@10 : {r10*100:.1f}%")

        # Latency (single query)
        lat = latency_profile(idx, query_embs, K)
        print(f"  P50={lat['p50_ms']}ms  P95={lat['p95_ms']}ms  P99={lat['p99_ms']}ms")

        # Energy for 1000 queries
        def run_1000():
            for i in range(1000):
                q = query_embs[i % len(query_embs)].reshape(1, -1)
                idx.search(q, K)

        _, energy_j, dur_s = monitor.measure(run_1000)
        print(f"  Energy (1000 queries): {energy_j:.2f} J  ({dur_s:.1f}s)")

        # Batch QPS
        qps = batch_qps(idx, query_embs, K, BATCH_SIZES)
        print(f"  QPS by batch size: { {k: f'{v:,.0f}' for k,v in qps.items()} }")

        results[name] = {
            "recall_at_10":         round(r10, 4),
            "latency":              lat,
            "energy_1000q_joules":  round(energy_j, 2),
            "qps_by_batch":         qps,
        }

    # ── nprobe sweep (IVF-PQ) ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print("nprobe sweep  →  IVF-PQ recall vs latency")
    print(f"{'='*55}")
    print(f"{'nprobe':>8}  {'Recall@10':>12}  {'P50 ms':>10}  {'Energy/1kQ J':>14}")
    sweep = []
    for nprobe in NPROBE_VALS:
        ivfpq_gpu.nprobe = nprobe
        _, pred = ivfpq_gpu.search(query_embs, K)
        r = recall_at_k(pred, gt, K)
        lat_n = latency_profile(ivfpq_gpu, query_embs, K, warmup=10, trials=100)
        def _q1k(): 
            for i in range(1000):
                ivfpq_gpu.search(query_embs[i % len(query_embs)].reshape(1, -1), K)
        _, e_j, _ = monitor.measure(_q1k)
        print(f"{nprobe:>8}  {r*100:>11.1f}%  {lat_n['p50_ms']:>9.2f}ms  {e_j:>13.2f}J")
        sweep.append({"nprobe": nprobe, "recall": round(r, 4),
                      "p50_ms": lat_n["p50_ms"], "energy_1kq_j": round(e_j, 2)})

    results["nprobe_sweep"] = sweep

    # ── Save ──────────────────────────────────────────────────────────
    out_path = res_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results → {out_path}")

    monitor.shutdown()


if __name__ == "__main__":
    main()
