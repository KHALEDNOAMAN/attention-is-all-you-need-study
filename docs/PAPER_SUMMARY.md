# Attention Is All You Need - Study Notes

## Key Concepts

### Self-Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

### Architecture
| Component | Purpose |
|-----------|--------|
| Multi-Head Attention | Attend to different representation subspaces |
| Feed-Forward Network | Non-linear transformation per position |
| Layer Normalization | Stabilize training |
| Positional Encoding | Inject sequence order information |
| Residual Connections | Gradient flow + identity shortcut |

### Why Transformers Won
- **Parallelizable**: Unlike RNNs, all positions processed simultaneously
- **Long-range dependencies**: Direct connections between any two positions
- **Scalable**: Scales better with data and compute

### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O
where head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
```

## Impact
- Foundation for BERT, GPT, T5, LLaMA, and all modern LLMs
- Revolutionized NLP, Computer Vision (ViT), and Audio
- Enabled scaling laws that drive current AI progress