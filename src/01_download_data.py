"""
01_download_data.py
────────────────────────────────────────────────────────────────
Downloads 200,000 MS-MARCO v1.1 passages and 500 dev queries.
Saves to:
  data/passages_200k.jsonl
  data/dev_queries.jsonl

Run:
  python src/01_download_data.py
  python src/01_download_data.py --gdrive  # saves to /content/drive/MyDrive/gpu-vector-search/
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

MAX_PASSAGES = 200_000
MAX_QUERIES  = 500


def get_data_dir(use_gdrive: bool) -> Path:
    if use_gdrive:
        d = Path("/content/drive/MyDrive/gpu-vector-search/data")
    else:
        d = Path("data")
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_passages(data_dir: Path):
    out_path = data_dir / "passages_200k.jsonl"
    if out_path.exists():
        print(f"[SKIP] {out_path} already exists.")
        return

    print(f"Streaming MS-MARCO passages (first {MAX_PASSAGES:,})...")
    dataset = load_dataset(
        "ms_marco", "v1.1", split="train",
        streaming=True, trust_remote_code=True
    )

    passages, passage_ids, count = [], [], 0
    for example in tqdm(dataset, total=MAX_PASSAGES, desc="Passages"):
        for text in example["passages"]["passage_text"]:
            passages.append(text)
            passage_ids.append(f"{example['query_id']}_{count}")
            count += 1
            if count >= MAX_PASSAGES:
                break
        if count >= MAX_PASSAGES:
            break

    with open(out_path, "w", encoding="utf-8") as f:
        for pid, text in zip(passage_ids, passages):
            f.write(json.dumps({"id": pid, "text": text}, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(passages):,} passages → {out_path}")


def download_queries(data_dir: Path):
    out_path = data_dir / "dev_queries.jsonl"
    if out_path.exists():
        print(f"[SKIP] {out_path} already exists.")
        return

    print(f"Loading {MAX_QUERIES} dev queries...")
    dev = load_dataset("ms_marco", "v1.1", split="validation", trust_remote_code=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for example in list(dev)[:MAX_QUERIES]:
            f.write(json.dumps({
                "id": str(example["query_id"]),
                "text": example["query"]
            }) + "\n")

    print(f"✓ Saved {MAX_QUERIES} dev queries → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive", action="store_true",
                        help="Save data to Google Drive mount")
    args = parser.parse_args()

    data_dir = get_data_dir(args.gdrive)
    download_passages(data_dir)
    download_queries(data_dir)
    print("\nData download complete.")
