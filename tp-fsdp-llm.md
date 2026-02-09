# Tensor Parallel + FSDP in LLM Training

## Glossary

- B: batch size
- S: input sequence length in tokens
- T: historical cache length in tokens
- D: hidden dimension
- H: number of heads
- N: number of GPU/TPUs
- d: D/H

## TP with Self-Attention

Parameter sharding:

- Q, K, V projection matrices are split along the output dimension (heads) across GPUs.
- The output projection matrix is split along its input dimension (also heads) across GPUs.

KV cache sharding:

- Split along heads. On each GPU: (B,T,H/N,d) for K and another (B,T,H/N,d) for V.

Steps:

1. The input (B,S,D) is replicated to all GPUs.
2. The multiplication of input with Q/K/V runs on all GPUs in parallel.  On each GPU, the result has shape (B,S,H/N,d),
3. Concat K and V. On each GPU, both K and V have shape (B,T+S,H/N,d).
4. On each GPU, compute attention matrix Q@K in parallel. The result has shape (B,S,T+S,H/N). The softmax noramlization applies on each of the H/N attention matrices independently. The multiplication with V also runs in parallel on all GPUs. The result is (B,S,H/N,d).
5. The input to the output projection is split along heads. The output projection parameter is also split along heads. The multiplication result needs to be all-reduced.  The final result replicated on all GPUs has the shape (B,S,D)

## TP with FFN

- The up projection is split along outputs (head dimension) like Q/K/V projections in self-attention.
- The down projection is split along inputs (also head dimension) like the output projection in self-attention.

## Plus FSDP

![](https://media.licdn.com/dms/image/v2/D4D22AQGFMcLCj1eG4Q/feedshare-shrink_800/B4DZWCAEUXG4Ak-/0/1741642811255?e=1772064000&v=beta&t=5ylHzFAYAv7997RjLp5QdJUhNyCIQdFYnULNv4s3PZM)

