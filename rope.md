# Efficient Implementation of Rotary Positional Embedding

Please refer to [my previous post](positional_encoding.html) for an explanation of why it is crucial to encode positional information for Transformer models. Additionally, [this post](https://medium.com/ai-insights-cobet/rotary-positional-embeddings-a-detailed-look-and-comprehensive-understanding-4ff66a874d83) offers a comparison of various approaches to encoding positional information. Rotary Positional Embedding (RoPE) is one of the best approaches.

## RoPE

RoPE takes a hidden state $x$ with the shape $(B, L, D)$, where $B$ is the batch size, $L$ is the sequence length, and $D$ is the hidden dimension. For simplicity, let's assume $B=1$.

RoPE encodes positional information by rotating the $l$-th column of $x$ by an angle $l \theta$. This rotational transformation is the source of the term "rotary" in its name.

Since the rotation is defined in 2D space, we must pair the $D$ elements in each column. Each pair is then rotated. To simplify implementation, we often "slice" the $L \times D$ matrix horizontally, treating $x_{l,i}$ and $x_{l,D/2+i}$ as a pair.

To distinguish between pairs in the same column, RoPE rotates each pair by different angles. For the $i$-th pair in the $l$-th column, the rotation angle is $l \theta_i$.

For simplicity, let's consider the case where $D=2$. In this scenario, the pair to be rotated is $(x_{l,0}, x_{l,1})$.

## Derivation 

[This post](https://www.cuemath.com/algebra/rotation-matrix/) provides a refresher on the rotation matrix and explains why rotating a pair $(x_0, x_1)$ results in:

$$( x_0 \cos\theta_l - x_1 \sin\theta_l, x_1 \cos\theta_l + x_0 \sin\theta_l ) $$

This inspires the following approach:

1. Precompute the $L \times D/2$ matrices: $\cos = \cos \theta_{l,i}$ and $\sin = \sin \theta_{l,i}$.
2. Rotate $x$ by first splitting it into two halves: the upper half $x^u$ (denoted by $x_0$ in the above equation) and the bottom half $x^b$ (denoted by $x_1$).

All four matrices—$\cos$, $\sin$, $x^u$, and $x^b$—have the same size, $L \times D/2$. Therefore, the result is the concatenation of the rotated upper and bottom halves:

$$
\begin{bmatrix}
x^u \circ \cos - x^d \circ \sin \\
x^d \circ \cos + x^u \circ \sin 
\end{bmatrix}
$$

This is equivalent to:

$$
x \circ \hat\cos + \hat{x} \circ \hat\sin
$$

where $\hat{x}=\begin{bmatrix}- x^d \\ x^u \end{bmatrix}$, $\hat\cos=\begin{bmatrix}\cos \\ \cos\end{bmatrix}$, and similarly $\hat\sin=\begin{bmatrix} \sin \\ \sin \end{bmatrix}$.

This derivation leads to the efficient RoPE implementation by Eleuther. Since we want the hidden dimension pairs to rotate at different rates, Eleuther's implementation creates a $D/2$-dimensional vector $\theta$:

$$
\theta = \begin{bmatrix}
\frac{1}{\text{base}^{0/D}} \\
\frac{1}{\text{base}^{2/D}} \\
\frac{1}{\text{base}^{4/D}} \\
\vdots \\
\end{bmatrix}
$$

RoPE rotates each position $l$ by the angle $l \theta_i$, giving us the $L \times D/2$ matrix $\Theta$:

$$ \Theta = \theta \times [0, 1, 2, \ldots, L]
= \begin{bmatrix}
\ldots & l \frac{1}{\text{base}^{0/D}} & \ldots \\
\ldots & l \frac{1}{\text{base}^{2/D}} & \ldots \\
\ldots & l \frac{1}{\text{base}^{4/D}} & \ldots \\
\vdots & \vdots                        & \vdots \\
\end{bmatrix}
$$

Then, we calculate:

$$
\hat\cos = \cos\left( \begin{bmatrix} \Theta \\ \Theta \end{bmatrix} \right)
$$

$$
\hat\sin = \sin\left( \begin{bmatrix} \Theta \\ \Theta \end{bmatrix} \right)
$$

## MLX Implementation

Eleuther provides this implementation in PyTorch, available at [their blog](https://blog.eleuther.ai/rotary-embeddings/). MLX offers an optimized implementation, `mlx.fast.rope`, in C++. Below is the MLX implementation in Python:

```python
import mlx.core as mx
from mlx import nn


class RoPE(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float16) / dim))  # θ

    def __call__(self, x):
        B, L, D = x.shape
        assert D == self.dim
        l = mx.arange(L).astype(self.theta.dtype)  # [0, 1, 2 ... L]
        Theta = mx.einsum("i,j->ij", l, self.theta)  # Θ
        hatTheta = mx.concatenate([Theta, Theta], axis=-1)
        sin = mx.sin(hatTheta) # Consider cache it. 
        cos = mx.cos(hatTheta)
        xu, xd = x[..., : D // 2], x[..., D // 2 :]
        hatx = mx.concatenate([-xd, xu], axis=-1)
        return x * cos + hatx * sin


B = 1
L = 3
D = 4
x = mx.ones([B, L, D])
rope = RoPE(dim=D)
o = rope(x)
print(o)
```
