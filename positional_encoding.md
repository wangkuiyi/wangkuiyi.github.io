# Why Transformer Models Need Positional Encoding

Yi Wang <yi dot wang dot 2005 在 Gmail>

Have you ever wondered why positional encoding is necessary in Transformer models? Yes, the original paper *Attention Is All You Need* provides an explanation, but it’s rather vague:

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.

Since the paper was published, there have been numerous tutorials discussing how to encode positional information. However, I haven't seen any that delve into the details. So, here is my attempt.

Consider the following: given a prompt consisting of two tokens, $[A, B]$, we are to generate a third token, $C$. This requires running the Transformer model, which contains several Transformer layers, and each layer consists of self-attention and a linear projection.

Let’s focus on the self-attention mechanism, which applies three linear projections to the input $x = [e(A); e(B), e(C)]$, where $e()$ denotes the embedding vector (or more generally, the hidden state) of a token. The outputs of these three projections are known as the queries, keys, and values of the input tokens, denoted by:

$$
q = \begin{bmatrix} q(C) \end{bmatrix} \;\;
k = \begin{bmatrix} k(A) \\ k(B) \\ k(C) \end{bmatrix} \;\;
v = \begin{bmatrix} v(A) \\ v(B) \\ v(C) \end{bmatrix}
$$

Note that during token generation (inference time), the query contains only one token, which differs from the training scenario where $q = [q(A) \\ q(B) \\ q(C)]$.

The definition of self-attention is as follows. It computes an attention matrix from $q$ and $k$, and uses it to weight the vectors in $v$. The $\text{softmax}$ function normalizes each row of the attention matrix:

$$ \text{softmax}(q \times k^T) \times v $$

In our case, the attention matrix is as follows:

$$
s = q \times k^T 
= [q(C)] \times \begin{bmatrix} k(A) \\ k(B) \\ k(C) \end{bmatrix}
= \begin{bmatrix} q(C) \cdot k(A) \\ q(C) \cdot k(B) \\ q(C) \cdot k(C) \end{bmatrix}
$$

The final output is:

$$
q(C) \cdot k(A) \; v(A) + q(C) \cdot k(B) \; v(B) + q(C) \cdot k(C) \; v(C)
$$

Now, let’s switch the order of tokens in the prompt from $[A, B]$ to $[B, A]$. Interestingly, the self-attention output is as follows, which has the exact same value as before because the summation operation does not depend on the order of its inputs:

$$
q(C) \cdot k(B) \; v(B) + q(C) \cdot k(A) \; v(A) + q(C) \cdot k(C) \; v(C)
$$
