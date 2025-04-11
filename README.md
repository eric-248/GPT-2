# 🧠 Reproducing GPT-2 (124M) from Scratch

This project is a from-scratch reproduction of the GPT-2 124M model, implemented by closely following the original [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/b) and incorporating insights from the [GPT-3 paper](https://arxiv.org/abs/2005.14165) and the ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper. It covers everything from model construction and training to optimization and evaluation.

---

## 🛠 Features

- PyTorch implementation of GPT-2 (124M) from scratch.
- Supports loading pretrained HuggingFace GPT-2 weights.
- End-to-end training loop (with overfitting a batch → full training).
- Mixed precision training, kernel fusion, and other speed-ups.
- Support for distributed training (DDP).
- Evaluation on HellaSwag and token sampling examples.
- Tokenizer-compatible with OpenAI GPT-2 (50257 → 50304 vocab).


---

## 🧠 Architecture Highlights

- Multi-head self-attention with causal masking
- GELU activation, LayerNorm, residual connections
- Learned positional embeddings
- Weight sharing between token embeddings (`wte`) and final `lm_head`
- Proper initialization: std 0.02, residual path scaled

---

## 🚀 Training Pipeline

### Step-by-Step Progression:
- ✅ Overfit a single batch with CrossEntropy loss
- ✅ DataLoader-lite for efficient batch feeding
- ✅ Full optimization loop with AdamW
- ✅ Learning rate scheduler (warmup + cosine decay)
- ✅ Batch size schedule, weight decay, gradient clipping
- ✅ Mixed precision (float16 / bfloat16), gradient scaling
- ✅ Kernel fusion with `torch.compile`
- ✅ FlashAttention integration
- ✅ Distributed training (DDP)
- ✅ Eval on validation set + real-world tasks

---

## 🚀 Acknowledgements

- OpenAI GPT-2 and GPT-3 Papers
- HuggingFace Transformers
- FlashAttention, Pytorch DDP
- Andrej Karpathy



