"""
src/benchmark.py
────────────────────────────────────────────────────────────────
Standalone benchmark script — can also be imported as a module.

Usage (from repo root after building indexes):
  python src/benchmark.py --idx-dir indexes --emb-dir embeddings --out benchmarks
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pynvml
import torch


# ─── NVML energy helper ──────────────────────────────────────────────────────

def _nvml_handle():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)

def energy_mj(handle) -> int:
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

def timed_energy(fn, handle):
    """Run fn(), return (result, elapsed_s, energy_joules)."""
    torch.cuda.synchronize()
    e0, t0 = energy_mj(handle), time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - t0, (energy_mj(handle) - e0) / 1000.0


# ─── Metrics ─────────────────────────────────────────────────────────────────

def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    return sum(len(set(p.tolist()) & set(g.tolist())) for p, g in zip(pred, gt)) / (len(gt) * k)


def latency_profile(idx, queries: np.ndarray, k: int,
                    warmup: int = 50, trials: int = 500) -> dict:
    is_gpu = hasattr(idx, "getDevice")
    lats = []
    for i in range(warmup + trials):
        q = queries[i % len(queries)].reshape(1, -1)
        if is_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx.search(q, k)
        if is_gpu:
            torch.cuda.synchronize()
        if i >= warmup:
            lats.append((time.perf_counter() - t0) * 1000)
    a = np.array(lats)
    return {
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
    }


def batch_qps(idx, queries: np.ndarray, k: int,
              batch_sizes: list[int] | None = None) -> dict:
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500, 1000]
    is_gpu = hasattr(idx, "getDevice")
    result = {}
    for bs in batch_sizes:
        batch = queries[:bs]
        for _ in range(3):
            idx.search(batch, k)
        if is_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            idx.search(batch, k)
        if is_gpu:
            torch.cuda.synchronize()
        result[str(bs)] = round((20 * bs) / (time.perf_counter() - t0), 1)
    return result


def energy_per_1k_queries(idx, queries: np.ndarray, k: int, handle) -> tuple[float, float]:
    def run():
        for i in range(1000):
            idx.search(queries[i % len(queries)].reshape(1, -1), k)
    _, dur, ej = timed_energy(run, handle)
    return round(ej, 2), round(dur, 2)


# ─── Main ────────────────────────────────────────────────────────────────────

def run_benchmark(idx_dir: Path, emb_dir: Path, out_dir: Path, k: int = 10):
    handle = _nvml_handle()
    res = faiss.StandardGpuResources()
    co  = faiss.GpuClonerOptions()
    co.use_cuvs = True

    query_embs   = np.load(emb_dir / "query_embs.npy").astype("float32")
    passage_embs = np.load(emb_dir / "passage_embs.npy").astype("float32")
    N, D = passage_embs.shape

    # Ground truth
    print("Computing brute-force ground truth...")
    flat = faiss.GpuIndexFlatIP(res, D)
    flat.add(passage_embs)
    torch.cuda.synchronize()
    _, GT = flat.search(query_embs, k)

    def load(fname, nprobe=None):
        cpu = faiss.read_index(str(idx_dir / fname))
        if "hnsw_cpu" in fname:
            cpu.hnsw.efSearch = 128
            return cpu
        g = faiss.index_cpu_to_gpu(res, 0, cpu, co)
        if nprobe:
            g.nprobe = nprobe
        return g

    indexes = {
        "CPU HNSW":          load("hnsw_cpu.faiss"),
        "GPU IVF-PQ (cuVS)": load("ivfpq_gpu.faiss", 16),
        "GPU CAGRA (cuVS)":  load("cagra_gpu.faiss"),
    }

    results = {}
    hdr = f"{'Index':<22} {'Recall@10':>10} {'P50ms':>7} {'P95ms':>7} {'P99ms':>7} {'E/1kQ J':>10}"
    print("\n" + hdr + "\n" + "-" * len(hdr))

    for name, idx in indexes.items():
        _, pred = idx.search(query_embs, k)
        r10     = recall_at_k(pred, GT, k)
        lat     = latency_profile(idx, query_embs, k)
        ej, _   = energy_per_1k_queries(idx, query_embs, k, handle)
        qps     = batch_qps(idx, query_embs, k)

        print(f"{name:<22} {r10*100:>9.1f}% {lat['p50_ms']:>7} {lat['p95_ms']:>7} {lat['p99_ms']:>7} {ej:>10}")
        results[name] = {
            "recall_at_10": round(r10, 4), "latency": lat,
            "energy_1000q_joules": ej, "qps_by_batch": qps,
        }

    # nprobe sweep
    print("\nnprobe sweep for IVF-PQ...")
    ivfpq = indexes["GPU IVF-PQ (cuVS)"]
    sweep = []
    for np_v in [1, 4, 8, 16, 32, 64, 128, 256]:
        ivfpq.nprobe = np_v
        _, pred = ivfpq.search(query_embs, k)
        r       = recall_at_k(pred, GT, k)
        lat_n   = latency_profile(ivfpq, query_embs, k, warmup=10, trials=100)
        ej, _   = energy_per_1k_queries(ivfpq, query_embs, k, handle)
        sweep.append({"nprobe": np_v, "recall": round(r,4),
                      "p50_ms": lat_n["p50_ms"], "energy_1kq_j": ej})
        print(f"  nprobe={np_v:<4} recall={r*100:.1f}%  p50={lat_n['p50_ms']}ms  E={ej}J")

    results["nprobe_sweep"] = sweep
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")
    pynvml.nvmlShutdown()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx-dir",  default="indexes",    type=Path)
    parser.add_argument("--emb-dir",  default="embeddings", type=Path)
    parser.add_argument("--out",      default="benchmarks", type=Path)
    parser.add_argument("--k",        default=10,           type=int)
    args = parser.parse_args()
    run_benchmark(args.idx_dir, args.emb_dir, args.out, args.k)
