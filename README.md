# Transformer

A from-scratch implementation of the Transformer model from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) in PyTorch, trained on the WMT14 English-German translation task.

## Project Structure

- `model.py` - Transformer architecture (Attention, Multi-Head Attention, Encoder, Decoder, Positional Encoding)
- `data.py` - Data loading and tokenization (word-level and BPE via SentencePiece)
- `train.py` - Training loop with warmup LR scheduler, early stopping, and wandb logging
- `masks.py` - Padding and causal mask utilities
- `inference.py` - Inference script

## Features

- Multi-head self-attention and cross-attention
- Positional encoding (sinusoidal)
- Learning rate warmup + decay schedule (from the original paper)
- Early stopping with best model checkpointing
- Experiment tracking with [Weights & Biases](https://wandb.ai)
- Dual tokenizer support: word-level and BPE (SentencePiece)
- Automatic WMT14 dataset download via HuggingFace Datasets

## Setup

```bash
pip install torch pandas sentencepiece datasets wandb
```

## Usage

### Training

```bash
wandb login
python train.py
```

The training script will:
1. Download WMT14 de-en dataset if not cached locally
2. Train SentencePiece BPE tokenizer (or use word-level with `tokenizer="word"`)
3. Train the Transformer model with warmup LR schedule
4. Log metrics to wandb
5. Save the best model to `best_model.pt`

### Configuration

Hyperparameters can be modified in the `config` dict in `train.py`:

```python
config = {
    "d_model": 768,
    "num_heads": 8,
    "d_ff": 2048,
    "num_layers": 6,
    "lr": 0.001,
    "epochs": 30,
    "patience": 5,
}
```
