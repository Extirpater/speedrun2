#!/usr/bin/env python3
"""
Stream‐merge, shard, shuffle & compress JSONL files from multiple subsets,
loading all parameters (including per‐subset valid sizes) from a JSON config.
"""

import json
import os
import random
import argparse
import zstandard as zstd
import shutil

def parse_args():
    p = argparse.ArgumentParser(
        description="Stream‐merge, shard, shuffle & compress jsonl datasets via a JSON config"
    )
    p.add_argument(
        "--config", "-c", required=True,
        help="Path to JSON config file"
    )
    return p.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    args        = parse_args()
    cfg         = load_config(args.config)
    input_root  = cfg["tmp_root"]
    output_root = cfg["out_root"]
    num_split   = cfg["num_split"]
    subsets     = cfg["datasets"]  # list of { "path": ..., "num_valid_samples": ... }

    train_dir = os.path.join(output_root, "train")
    valid_dir = os.path.join(output_root, "valid")
    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(valid_dir, exist_ok=False)

    # Prepare zstd stream writers for each train shard
    train_writers = []
    for i in range(num_split):
        out_path = os.path.join(train_dir, f"train_{i}.jsonl.zst")
        f        = open(out_path, "wb")
        compressor = zstd.ZstdCompressor().stream_writer(f)
        train_writers.append((f, compressor))

    for entry in subsets:
        subpath = entry["path"]
        n_valid = entry["num_valid_samples"]
        src_file = os.path.join(input_root, subpath, "train.jsonl")

        print(f"\nProcessing subset '{subpath}'")

        # 1) Count total lines
        total = 0
        with open(src_file, "r") as fin:
            for _ in fin:
                total += 1

        n_train   = total - n_valid
        per_shard = n_train // num_split
        used      = per_shard * num_split + n_valid
        dropped   = total - used

        print(f"  total={total}, used={used} (dropped={dropped}), train={n_train}, valid={n_valid}")

        # 2) Pick validation indices in [0, used)
        valid_idxs = set(random.sample(range(used), n_valid))

        # 3) Setup per-shard remaining counters
        remaining    = [per_shard] * num_split
        valid_buffer = []

        # 4) Stream lines into train shards or valid buffer
        with open(src_file, "r") as fin:
            for idx, line in enumerate(fin):
                if idx >= used:
                    break

                if idx in valid_idxs:
                    valid_buffer.append(line)
                else:
                    # pick a random shard that still needs lines
                    choices = [i for i, r in enumerate(remaining) if r > 0]
                    sid     = random.choice(choices)
                    _, comp = train_writers[sid]
                    comp.write(line.encode("utf-8"))
                    remaining[sid] -= 1

        # 5) Shuffle & write this subset’s valid file
        random.shuffle(valid_buffer)
        valid_out = os.path.join(valid_dir, f"{subpath}.jsonl.zst")
        os.makedirs(os.path.dirname(valid_out), exist_ok=True)  # ← ensure parent exists
        with open(valid_out, "wb") as vf:
            vc = zstd.ZstdCompressor().stream_writer(vf)
            for ln in valid_buffer:
                vc.write(ln.encode("utf-8"))
            vc.close()

        print(f"  → subset '{subpath}': wrote {n_train} train lines and {n_valid} valid lines")

    # 6) Close all train shard writers
    for f, comp in train_writers:
        comp.close()
        f.close()

    # 7) Clean up
    if os.path.isdir(input_root):
        print(f"\nAll done. Deleting temporary directory: {input_root}")
        shutil.rmtree(input_root)
    else:
        print(f"\nInput directory not found, skipping deletion: {input_root}")

if __name__ == "__main__":
    main()


