#!/usr/bin/env bash
# run_concat.sh
#
# Usage: ./run_concat.sh [CONFIG_DOWN] [CONFIG_SHARD] [LOG_FILE]
# Defaults:
#   CONFIG_DOWN  = configs/test.json
#   CONFIG_SHARD = configs/test_shard.json
#   LOG_FILE     = data.out

dir="./test_dataset"
if [[ -d "$dir" ]]; then
  rm -rf -- "$dir"
fi

set -e

CONFIG=${1:-configs/data.json}
LOG_FILE=${2:-data.out}

echo "Starting token concatenation with config: $CONFIG"
# echo "Logging to: $LOG_FILE"
echo "No nohup"

# Run download_and_tokenize, redirecting stdout/stderr to log
python scripts/download_and_tokenize.py --config "$CONFIG"

echo "download_and_tokenize completed. Starting sharding with config: $CONFIG"

python scripts/shard.py --config "$CONFIG"
