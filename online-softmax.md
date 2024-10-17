# FlashAttention (Part 2): Online Softmax

This is the second part of my reading notes of [Zihao Ye's note on FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf).

The definition of softmax is as follows:

$$\left\{ \frac{\exp(x_i)}{\sum_{j=1}^N \exp(x_j)} \right\}_{i=1}^N$$

It is well-known numerically instable -- if any $x_i\geq 11$, $\exp(x_i)$ exceeds the maximum value of float16. To address this, we compute an alternative form which gives equivalent result but numerically stable:

$$ \left\{ \frac{\exp(x_i-m)}{\sum_{j=1}^N \exp(x_j-m)} \right\}_{i=1}^N $$

where $m=\max_{j=1}^N x_j$.  This form is safe because $x_i - m \leq 0$, ensuring that $0 < \exp(x_i - m) \leq 1$.

Given an input array ${x_i}{i=1}^N$, the traditional algorithm proceeds by performing the following three inductive processes sequentially to compute the result array ${a_i}{i=1}^N$:

$$
\begin{aligned}
m_1 \quad & \ldots & m_i=\max(m_{i-1}, x_m) \quad        & \ldots & m_N \\
d_1 \quad & \ldots & d_i=d_{i-1} +\exp(x_i-m_N) \quad    & \ldots & d_N \\
a_1 \quad & \ldots & a_i=\frac{\exp(x_i-m_N)}{d_N} \quad & \ldots & a_N \\
\end{aligned}
$$

However, we prefer inductive processes that can run in parallel on a GPU. This would allow us to load the array ${x_i}{i=1}^N$ once and save the result ${a_i}{i=1}^N$ without needing to store or load intermediate results like ${m_i}$ and ${d_i}$. Unfortunately, the three processes above cannot run in parallel because $d_i$ depends on $m_N$, and $a_i$ depends on both $d_N$ and $m_N$.

To address this, let’s explore whether we can construct a surrogate of $d_i$, denoted as $\delta_i$, that allows the inductive processes for $m_i$ and $\delta_i$ to run in parallel. Specifically, we want $\delta_i$ to satisfy the following properties:

1. We want $\delta_i = \sum_{j=1}^i \exp(x_j - m_i)$, so that $\delta_N = \sum_{j=1}^N \exp(x_j - m_N)$, which is required to compute $a_i$.
2. Since $\delta_i$ is inductive, it should depend on $\delta_{i-1}$.
3. To allow parallel execution, $\delta_i$ must not depend on future values such as $x_{i+1}, \ldots$ or $m_{i+1}, \ldots$.

We begin by considering:

$$\delta_i=\sum_{j=1}^i \exp(x_j-m_i)$$

To ensure $\delta_i$ depends on $\delta_{i-1}$, which is:

$$\delta_{i-1}=\sum_{j=1}^{i-1} \exp(x_j-m_{i-1})$$

we need to split $\delta_i$ into two parts: one involving $\delta_{i-1}$ (which should not depend on $x_i$ or $m_i$), and the remaining terms that depend on $x_i$ and $m_i$. The first step is straightforward -- we separate the last term in the summation:

$$\delta_i=\sum_{j=1}^{i-1} \exp(x_j-m_i) + \exp(x_i-m_i)$$

Now, $x_i$ only appears in the second term. However, $m_i$ still appears in the summation. Let’s take the next step:

$$
\begin{aligned}
\delta_i &= \sum_{j=1}^{i-1} \exp(x_j-m_{i-1}+m_{i-1}-m_i) + \exp(x_i-m_i) \\
         &= \left[\sum_{j=1}^{i-1} \exp(x_j-m_{i-1})\right] \exp(m_{i-1}-m_i) + \exp(x_i-m_i)
\end{aligned}
$$

The expression inside the square brackets is exactly $\delta_{i-1}$. Therefore, we have:

$$
\delta_i = \delta_{i-1} \exp(m_{i-1}-m_i) + \exp(x_i-m_i)
$$

This allows us to compute $\delta_i$ inductively in parallel with $m_i$.
