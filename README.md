# attentif

[![codecov](https://codecov.io/gh/neet/attentif/graph/badge.svg?token=E4x4ZETQ7S)](https://codecov.io/gh/neet/attentif)

A toy implementation of ”Attention Is All You Need”

![A matplotlib capture for loss vs step](./assets/thumbnail.png)

## Demo

### BERT

![Screenshot of Jupyter Lab, solving a fill-mask task by BERT](./assets/bert.jpeg)

## Motivation

I made this project in order to get a deeper understanding for the Transformer architecture, BERT, RoBERTa, T5, and GPT models. We often rely on existing Transformer implementation when we need to train a model, but I wanted to test if I can implement them from scratch, referring to the paper.

This project **does** include:
- `torch.nn.Module`
- `torch.nn.Parameter`
- Existing tokenizer implementation from `transformers`
- and other primitive functions offered by PyTorch

While this project **does not** include:
- Models from `transformers`
- `nn.Transformer`
- `nn.MultiHeadAttention`
- `nn.Embedding`
- `nn.LayerNorm`
- `nn.functional.softmax`
- and other existing modules that plays an essential role in Transformer architecture

## Features

We implemented the following features so far. You can find the layers and functions in `src/layers`, and models in `src/models`.

### Functions

- [x] `dropout`
- [x] `softmax`
- [x] `relu`
- [ ] `gelu`
- [x] `positional_encoding`

### Layers

- [x] `MultiHeadAttention`
- [x] `FeedForwardNetwork`
- [x] `LayerNorm`
- [x] `TokenEmbedding`
- [x] `TransformerEncoder`
- [x] `TransformerEncoderBlock`
- [ ] `TransformerDecoder`
- [ ] `TransformerDecoderBlock`

### Models

- [x] `BertModel`
- [ ] `GPT2Model`
- [ ] `T5Model`

### Schedulers

We use `transformers` for schedulers for now, but have a plan to implement them from scratch in the future.

- [ ] `AdamW`
- [ ] `CrossEntropy`

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). NAACL 2019.

