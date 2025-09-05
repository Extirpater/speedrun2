# Speedrun: Efficient Language Model Training

A high-performance framework for training and evaluating language models with advanced optimization techniques, including custom optimizers and quantization methods.

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (recommended: H100)
- Conda or pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd speedrun
   ```

2. **Set up the environment:**
   ```bash
   bash setup.sh
   ```

3. **Prepare the dataset:**
   ```bash
   cd data
   bash process_data.sh
   cd ..
   ```

4. **Configure Weights & Biases (optional but recommended):**
   ```bash
   wandb login
   ```

5. **Start training:**
   ```bash
   bash run.sh
   ```

6. **Evaluate the model:**
   ```bash
   bash eval.sh
   ```

## 📁 Project Structure

```
speedrun/
├── alg/                          # Core training algorithms
│   ├── args.py                   # Training configuration and arguments
│   ├── run.py                    # Main training entry point
│   ├── merge_trainer.py          # Custom trainer with advanced features
│   ├── cadamw.py                 # Custom AdamW optimizer implementation
│   ├── muon.py                   # Muon optimizer implementation
│   ├── models.py                 # Model definitions and utilities
│   ├── data.py                   # Data loading and preprocessing
│   ├── metrics.py                # Training metrics and evaluation
│   └── objectives/               # Loss functions and training objectives
│       ├── loss.py               # Core loss functions
│       ├── objectives.py         # Training objective implementations
│       ├── projectors.py         # Feature projection layers
│       ├── whiten.py             # Whitening transformations
│       ├── norm.py               # Normalization utilities
│       └── layer_mappers.py      # Layer mapping utilities
├── data/                         # Data processing and configuration
│   ├── configs/
│   │   └── data.json             # Dataset configuration
│   ├── scripts/                  # Data processing scripts
│   └── process_data.sh           # Data preparation script
├── eval.py                       # Model evaluation script
├── requirements.txt              # Python dependencies
├── setup.sh                      # Environment setup script
├── run.sh                        # Training script
└── eval.sh                       # Evaluation script
```

## ⚙️ Configuration

### Training Parameters

Key training parameters can be modified in `alg/args.py`:

- **Model**: `model_name` - HuggingFace model identifier
- **Optimizer**: `optimizer_name` - Choice of optimizer (cadamw, muon, etc.)
- **Batch Size**: `per_device_train_batch_size` - Training batch size per device
- **Learning Rate**: `learning_rate` - Initial learning rate
- **Max Steps**: `max_steps` - Total training steps
- **Context Length**: `concat_tokens` - Sequence length for training

### Dataset Configuration

Modify `data/configs/data.json` to configure your training datasets:

```json
{
  "datasets": [
    { "path": "dataset_name", "limit": 0.51, "num_valid_samples": 8 },
    // Add more datasets...
  ],
  "concat_tokens": 4096,
  "tokenizer": "kaizen9/test2b12"
}
```

Each dataset entry includes:
- `path`: HuggingFace dataset identifier
- `limit`: Fraction of total tokens (0.0-1.0)
- `num_valid_samples`: Number of validation samples

## 🔧 Key Features

### Advanced Optimizers
- **CaAdamW**: Custom AdamW implementation with enhanced features
- **Muon**: Novel optimizer with auxiliary parameters
- **Liger Kernel**: Optional kernel optimization for performance

### Quantization Support
- **TWN Quantization**: Ternary Weight Networks for model compression
- **Flash Attention**: Optimized attention implementation
- **Gradient Checkpointing**: Memory-efficient training

### Training Features
- **Knowledge Distillation**: Student-teacher training paradigm
- **Multi-GPU Support**: Distributed training with Accelerate
- **Weights & Biases Integration**: Comprehensive experiment tracking
- **Gradient Statistics**: Advanced gradient analysis and monitoring

## 📊 Evaluation

The framework includes comprehensive evaluation on standard benchmarks:

- **MMLU**: Massive Multitask Language Understanding
- **HellaSwag**: Commonsense reasoning
- **ARC Challenge**: AI2 Reasoning Challenge
- **Custom Tasks**: Extensible evaluation framework

### Running Evaluations

```bash
# Evaluate on all default tasks
python eval.py --model_name path/to/model

# Evaluate on specific tasks
python eval.py --model_name path/to/model --tasks mmlu hellaswag

# Few-shot evaluation
python eval.py --model_name path/to/model --num_fewshot 5
```

## 🎯 Performance

Baseline results achieved with default configuration:

| Benchmark | 0-shot | 5-shot |
|-----------|--------|--------|
| MMLU | 44.08% | 46.30% |
| HellaSwag | 51.99% | - |
| ARC Challenge | 36.00% | - |

## 🛠️ Advanced Usage

### Custom Training

```python
from alg.run import train
from alg.args import get_args

# Get default arguments
training_args, model_args, dataset_args, eval_args = get_args()

# Modify arguments as needed
training_args.learning_rate = 2e-4
training_args.max_steps = 20000

# Start training
train(training_args, model_args, dataset_args, eval_args)
```

### Benchmarking

```python
from alg.run import benchmark

# Run hyperparameter sweep
benchmark(
    learning_rate=[1e-4, 2e-4, 4e-4],
    optimizer_name=["cadamw", "muon"],
    per_device_train_batch_size=[2, 4, 8]
)
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `per_device_train_batch_size` or enable gradient checkpointing
2. **Data Loading Errors**: Decrease `num_workers` in data processing scripts
3. **Slow Training**: Enable `torch_compile` and ensure proper GPU utilization

### Performance Tips

- Use `bf16` precision for better performance on modern GPUs
- Enable gradient checkpointing for memory efficiency
- Use appropriate batch sizes for your hardware
- Monitor GPU utilization and adjust accordingly

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📚 References

- [AdamW Paper](https://arxiv.org/abs/1711.05101)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)