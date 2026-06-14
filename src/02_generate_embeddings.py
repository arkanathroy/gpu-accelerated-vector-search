"""
02_generate_embeddings.py
────────────────────────────────────────────────────────────────
Encodes passages and queries using all-MiniLM-L6-v2 (384-dim).
L2-normalises vectors so inner product equals cosine similarity.

Saves to embeddings/:
  passage_embs.npy   shape (200000, 384)  float32
  query_embs.npy     shape (500, 384)     float32

Run:
  python src/02_generate_embeddings.py
  python src/02_generate_embeddings.py --gdrive
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512   # T4 has 16 GB VRAM; 512 is safe for this model (~90 MB)


def get_dirs(use_gdrive: bool):
    base = Path("/content/drive/MyDrive/gpu-vector-search") if use_gdrive else Path(".")
    data_dir  = base / "data"
    emb_dir   = base / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, emb_dir


def load_texts(jsonl_path: Path) -> list[str]:
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def encode_and_save(model, texts: list[str], out_path: Path, desc: str):
    if out_path.exists():
        print(f"[SKIP] {out_path} already exists.")
        arr = np.load(out_path)
        print(f"  Loaded: {arr.shape}")
        return arr

    print(f"\nEncoding {len(texts):,} {desc}...")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-norm → cosine via inner product
    ).astype("float32")

    np.save(out_path, embs)
    print(f"✓ Saved {embs.shape} → {out_path}")
    return embs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive", action="store_true")
    args = parser.parse_args()

    data_dir, emb_dir = get_dirs(args.gdrive)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim   = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")   # 384

    passages = load_texts(data_dir / "passages_200k.jsonl")
    queries  = load_texts(data_dir / "dev_queries.jsonl")

    encode_and_save(model, passages, emb_dir / "passage_embs.npy", "passages")
    encode_and_save(model, queries,  emb_dir / "query_embs.npy",   "queries")

    # VRAM preview for user
    raw_mb   = len(passages) * dim * 4 / 1024**2
    ivfpq_mb = len(passages) * 32   / 1024**2
    print(f"\nVRAM estimate for 200K × {dim}-dim vectors on T4:")
    print(f"  Raw float32          : {raw_mb:>7.1f} MB")
    print(f"  IVF-PQ (M=32 bytes)  : {ivfpq_mb:>7.1f} MB")
    print(f"  T4 VRAM available    :  16,384 MB  ✓ fits easily")
