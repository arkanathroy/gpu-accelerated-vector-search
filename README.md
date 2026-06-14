# ⚡ GPU-Accelerated Vector Search Service

A production-grade semantic search microservice over **200K MS-MARCO passages** powered by FAISS + NVIDIA cuVS (CAGRA and IVF-PQ), with a gRPC API and full benchmark suite.

> **Portfolio project** demonstrating GPU programming skills: custom index kernels, NVML energy profiling, async gRPC serving, and quantitative benchmarking against CPU baseline.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Client (Python gRPC stub)                                     │
└───────────────────────┬────────────────────────────────────────┘
                        │ gRPC (proto3)
┌───────────────────────▼────────────────────────────────────────┐
│  SearchService  (src/05_grpc_server.py)                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  IndexStore — loads all indexes at startup             │    │
│  │  ┌──────────┐  ┌─────────────────┐  ┌──────────────┐   │    │
│  │  │ CPU HNSW │  │ GPU IVF-PQ cuVS │  │ GPU CAGRA    │   │    │
│  │  │ baseline │  │ 48x compressed  │  │ cuVS highest │   │    │
│  │  └──────────┘  └─────────────────┘  └──────────────┘   │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────────┐
│  FAISS + NVIDIA cuVS (CUDA CC≥7.0)                             │
│  NVML Energy Profiling │ sentence-transformers embeddings      │
└────────────────────────────────────────────────────────────────┘
```

## Quick Start — Google Colab (T4)

1. Open `notebooks/gpu_vector_search_colab.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells top to bottom

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arkanathroy/gpu-accelerated-vector-search/blob/main/notebooks/gpu_vector_search_colab.ipynb)

---

## Repository Structure

```
gpu-vector-search/
├── notebooks/
│   └── gpu_vector_search_colab.ipynb   # Full end-to-end Colab notebook
├── src/
│   ├── benchmark.py                    # Standalone benchmark module
│   └── 05_grpc_server.py               # Async gRPC server
├── proto/
│   └── search.proto                    # Protobuf service definition
├── tests/
│   └── test_indexes.py                 # CPU sanity tests (CI)
├── .github/workflows/
│   └── ci.yml                          # Lint + test on every push
├── requirements.txt
└── README.md
```

## Key Technical Decisions

| Decision                                          | Why                                                                                    |
| ------------------------------------------------- | -------------------------------------------------------------------------------------- |
| CAGRA `train()` only                              | CAGRA indexes all data during `train()`; calling `add()` is wrong and raises an error  |
| `index_gpu_to_cpu()` before saving                | GPU indexes cannot be serialized directly; must convert to CPU first                   |
| RMM pool allocator                                | Reduces CUDA malloc overhead on repeated small allocations                             |
| L2-normalised embeddings + `METRIC_INNER_PRODUCT` | Inner product on unit vectors = cosine similarity; avoids computing `METRIC_L2` + sqrt |
| `torch.cuda.synchronize()` before NVML reads      | Ensures all GPU work is complete before reading energy counters                        |

## Benchmark Results (T4 GPU, 200K vectors, 384-dim, K=10)

| Index               | Recall@10 | P50 Latency | Energy/1kQ |
| ------------------- | --------- | ----------- | ---------- |
| CPU HNSW (baseline) | ~98%      | ~8ms        | N/A (CPU)  |
| GPU IVF-PQ (cuVS)   | ~85%      | ~2ms        | ~1.2J      |
| GPU CAGRA (cuVS)    | ~96%      | ~1ms        | ~0.8J      |

> Actual numbers from your run will be in `benchmarks/results.json`.

## Hardware Requirements

| Component | Minimum             | Tested        |
| --------- | ------------------- | ------------- |
| GPU       | CUDA CC 7.0 (Volta) | T4 (CC 7.5)   |
| VRAM      | 4 GB                | 16 GB         |
| CUDA      | 12.x                | 12.4          |
| RAM       | 16 GB               | 52 GB (Colab) |

> GTX 1050 Ti (CC 6.1) does **not** meet the CC≥7.0 requirement for cuVS.

## Running Tests (CPU)

```bash
pip install pytest faiss-cpu numpy
pytest tests/ -v
```

## License

MIT
