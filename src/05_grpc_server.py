"""
05_grpc_server.py
────────────────────────────────────────────────────────────────
Async gRPC server exposing SearchService and HealthCheck RPCs.

Loads all three FAISS indexes + sentence-transformer at startup.
Routes search requests to the requested index (default: cagra).

Run (after protoc generation):
  python src/05_grpc_server.py
  python src/05_grpc_server.py --gdrive --port 50051
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import faiss
import grpc
import numpy as np
import pynvml
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "generated")
import search_pb2
import search_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Index Store ─────────────────────────────────────────────────────────────

class IndexStore:
    """Loads all indexes and the embedding model once at startup."""

    def __init__(self, base_dir: Path):
        idx_dir  = base_dir / "indexes"
        data_dir = base_dir / "data"

        pynvml.nvmlInit()
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        log.info("Initialising GPU resources...")
        self.res = faiss.StandardGpuResources()
        co       = faiss.GpuClonerOptions()
        co.use_cuvs = True

        log.info("Loading passage text store...")
        self.passages   = {}
        self.passage_ids = []
        with open(data_dir / "passages_200k.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.passages[obj["id"]] = obj["text"]
                self.passage_ids.append(obj["id"])

        log.info("Loading CPU HNSW index...")
        self.hnsw = faiss.read_index(str(idx_dir / "hnsw_cpu.faiss"))
        self.hnsw.hnsw.efSearch = 128

        log.info("Loading GPU IVF-PQ (cuVS) index...")
        ivfpq_cpu  = faiss.read_index(str(idx_dir / "ivfpq_gpu.faiss"))
        self.ivfpq = faiss.index_cpu_to_gpu(self.res, 0, ivfpq_cpu, co)
        self.ivfpq.nprobe = 16

        log.info("Loading GPU CAGRA (cuVS) index...")
        cagra_cpu  = faiss.read_index(str(idx_dir / "cagra_gpu.faiss"))
        self.cagra = faiss.index_cpu_to_gpu(self.res, 0, cagra_cpu, co)

        self.index_map = {
            "hnsw":   self.hnsw,
            "ivfpq":  self.ivfpq,
            "cagra":  self.cagra,
        }

        log.info("Loading sentence-transformer (all-MiniLM-L6-v2)...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        gpu_name = pynvml.nvmlDeviceGetName(self._nvml_handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        log.info(
            f"Server ready | GPU={gpu_name} | "
            f"VRAM used={mem_info.used/1024**3:.1f}/{mem_info.total/1024**3:.1f} GB"
        )

    def gpu_mem_info(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        return info.used / 1024**3, info.total / 1024**3

    def gpu_name(self) -> str:
        return pynvml.nvmlDeviceGetName(self._nvml_handle)


# ─── gRPC Servicer ───────────────────────────────────────────────────────────

class SearchServicer(search_pb2_grpc.SearchServiceServicer):

    def __init__(self, store: IndexStore):
        self.store = store

    async def Search(self, request, context):
        index_key = request.index_type if request.index_type in self.store.index_map else "cagra"
        k         = max(request.top_k, 1)

        q_vec = self.store.encoder.encode(
            [request.query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        index = self.store.index_map[index_key]
        t0    = time.perf_counter()
        distances, indices = index.search(q_vec, k)
        latency_ms = (time.perf_counter() - t0) * 1000

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.store.passage_ids):
                continue
            pid  = self.store.passage_ids[int(idx)]
            text = self.store.passages.get(pid, "")
            results.append(search_pb2.SearchResult(
                passage_id   = pid,
                passage_text = text[:500],
                score        = float(dist),
                rank         = rank,
            ))

        return search_pb2.SearchResponse(
            results    = results,
            latency_ms = float(latency_ms),
            index_used = index_key,
        )

    async def HealthCheck(self, request, context):
        used_gb, total_gb = self.store.gpu_mem_info()
        return search_pb2.HealthResponse(
            status          = "ok",
            vectors_indexed = self.store.cagra.ntotal,
            gpu_name        = self.store.gpu_name(),
            vram_used_gb    = float(used_gb),
            vram_total_gb   = float(total_gb),
        )


# ─── Entry Point ─────────────────────────────────────────────────────────────

async def serve(base_dir: Path, port: int):
    store  = IndexStore(base_dir)
    server = grpc.aio.server()
    search_pb2_grpc.add_SearchServiceServicer_to_server(SearchServicer(store), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    log.info(f"gRPC server listening on {listen_addr}")
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive", action="store_true")
    parser.add_argument("--port",   type=int, default=50051)
    args = parser.parse_args()

    base = Path("/content/drive/MyDrive/gpu-vector-search") if args.gdrive else Path(".")
    asyncio.run(serve(base, args.port))
