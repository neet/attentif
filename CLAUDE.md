# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

attentif is a toy implementation of the "Attention Is All You Need" paper, built from scratch using PyTorch. All components (attention, layer norm, feed-forward networks, etc.) are implemented manually without relying on `torch.nn.TransformerEncoder` or similar high-level abstractions.

## Development Environment

This project uses **uv** as the package manager (not pip/poetry). Python 3.13+ is required.

### Essential Commands

```bash
# Install dependencies
uv sync --all-extras --dev

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=xml

# Run a single test file
uv run pytest src/multi_head_attention_test.py

# Run a specific test function
uv run pytest src/multi_head_attention_test.py::test_function_name

# Lint with Ruff
uv run ruff check .

# Type check with Mypy
uv run mypy .

# Start Jupyter Lab (for notebooks)
uv run jupyter lab
```

## Architecture Overview

### Core Transformer Components

The project implements a standard transformer encoder architecture with all components built from scratch:

1. **MultiHeadAttention** (`src/multi_head_attention.py`):
   - Implements scaled dot-product attention with multiple heads
   - Uses learnable weight matrices W_Q, W_K, W_V, W_O (and biases b_Q, b_K, b_V, b_O)
   - Accepts optional `attention_mask` parameter for padding/causal masking
   - Mask format: (batch_size, seq_len, seq_len) tensor with 0.0 for valid positions, -inf for masked positions

2. **TransformerEncoderBlock** (`src/transformer_encoder_block.py`):
   - Follows pre-LN (pre-layer normalization) architecture
   - Structure: LayerNorm → MultiHeadAttention → Residual → LayerNorm → FFN → Residual
   - Includes dropout after both attention and FFN sublayers

3. **TransformerEncoder** (`src/transformer_encoder.py`):
   - Stacks multiple `TransformerEncoderBlock` instances
   - Applies final layer normalization after all blocks

4. **MaskedLM** (`src/mini_bert.py`):
   - Complete BERT-style masked language model
   - Uses weight tying between token embeddings and LM head
   - Default config: hidden_size=512, num_attention_heads=8, num_blocks=12
   - Handles attention mask conversion from (batch_size, seq_len) to (batch_size, seq_len, seq_len) format

### Key Implementation Details

- **Positional Encoding** (`src/positional_encoding.py`): Uses sinusoidal positional encoding (not learned). Returns (seq_len, hidden_size) tensor that broadcasts with batch.

- **Attention Masks**:
  - `make_padding_mask()` in `src/mask_padding.py`: Converts token IDs to mask (0.0 for valid, -inf for padding)
  - `make_causal_mask()` in `src/mask_causal.py`: Creates lower-triangular mask for autoregressive models
  - Masks are added to attention scores before softmax

- **Custom Implementations**: All basic operations (softmax, relu, dropout, layer_norm) are implemented from scratch in separate modules for educational purposes.

- **Weight Initialization**:
  - Xavier/Glorot uniform for attention and FFN weight matrices
  - Normal distribution (std=0.02) for embedding matrices (following RoBERTa convention)

### Testing Structure

- Each module has a corresponding `*_test.py` file
- Tests use PyTorch's testing utilities and cover shape transformations, gradient flow, and numerical correctness
- Test files follow the naming convention: `module_name_test.py`

### Notebooks

Training and experimentation notebooks are located in `notebooks/`. Use `uv run jupyter lab` to access them.

## Important Conventions

- **Variable Naming**:
  - Use descriptive names instead of single-letter acronyms:
    - `batch_size`: Batch size
    - `seq_len`: Sequence length (max_length)
    - `hidden_size`: Hidden dimension of attention/transformer
    - `num_attention_heads`: Number of attention heads
    - `attention_head_size`: Per-head key/value dimensions (typically `hidden_size // num_attention_heads`)
    - `intermediate_size`: FFN intermediate dimension (expansion before ReLU)
    - `vocab_size`: Vocabulary size
    - `embedding`: Embedding matrix
    - `prob`: Probability (e.g., dropout rate)
    - `weight`: Linear transformation weight matrix
    - `bias`: Linear transformation bias vector

- **Tensor Shapes**: Shape comments are included throughout the codebase (e.g., `# (batch_size, seq_len, hidden_size)`). Preserve these when editing.

- **Device/Dtype Handling**: Functions that create new tensors accept `device` and `dtype` parameters to match input tensors. Always propagate these when working with positional encodings or masks.

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yaml`) runs:
1. Ruff (continues on error)
2. Mypy (must pass)
3. Pytest with coverage (must pass)
4. Codecov upload
