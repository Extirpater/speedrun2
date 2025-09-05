#!/bin/bash

# Speedrun Evaluation Script
# This script evaluates a trained model on standard benchmarks

set -e  # Exit on any error

# Default configuration
MODEL_NAME="${1:-test_model/checkpoint-16000}"
TASKS="${2:-mmlu hellaswag arc_challenge}"

echo "Evaluating model: $MODEL_NAME"
echo "Tasks: $TASKS"

# Run evaluation
python eval.py --model_name "$MODEL_NAME" --tasks $TASKS

echo "Evaluation completed!"