# JAX's vmap: An Application of Speculative Decoding

In our recent work on a novel speculative decoding approach tailored for Apple Intelligence, we prototyped the idea using [PyTorch](https://github.com/apple/ml-recurrent-drafter/blob/main/recurrent_drafting/modeling_drafter.py).  Programming the vectorized algorithm that samples token sequences from a draft model posed a slight challenge.  We had to consider that for each prompt, we would sample a beam of multiple token sequences, and the input would be a batch of multiple prompts.

This inspires me to attempt writing the non-vectorized and comprehensible version of the code in JAX and using [JAX's vmap](vmap.html) to automatically add the batch and beam dimensions.  The trial was successful, as illustrated by the following code:

```python
from typing import Tuple

import jax
import jax.numpy as jnp

d = 8  # hidden dim of llm
h = 4  # hidden dim of rnn
l = 3  # number of tokens drawn from rnn after the one from llm
w = 5  # beam width
b = 2  # batch size


def rnn_drafter(llm_state, rnn_state, input_token) -> Tuple[jax.Array, jax.Array]:
    """[d], [h], [] -> [h], []"""
    print(f"\nrnn_state=\n{rnn_state}")
    rnn_state = rnn_state + input_token
    return rnn_state, input_token + rnn_state.sum()


def sample_seq(llm_state, llm_token) -> jax.Array:
    """[h], [] -> [l]"""
    token = llm_token
    rnn_state = jnp.zeros((h,))
    seq = jnp.array([token], dtype=jnp.int32)
    for _ in range(l):
        rnn_state, token = rnn_drafter(llm_state, rnn_state, token)
        seq = jnp.append(seq, token)
    return seq


def test_sample_seq():
    llm_state, llm_token = jnp.zeros((d,)), jnp.array([100], dtype=jnp.int32)
    assert jnp.all(sample_seq(llm_state, llm_token) == jnp.arange(100, 100 + l + 1))


def test_sample_beam():
    llm_state, llm_token = jnp.zeros((d,)), jnp.arange(0, w * 100, 100)
    print(jax.vmap(sample_seq, in_axes=(None, 0), out_axes=0)(llm_state, llm_token))


def test_sample_beam_batched():
    llm_state, rnn_state = (
        jnp.zeros((b, d)),
        jnp.zeros((b, w, h)),
    )
    llm_token = jnp.tile(jnp.arange(0, w * 100, 100), (b, 1))
    print(
        jax.vmap(jax.vmap(sample_seq, in_axes=(None, 0), out_axes=0), in_axes=(0, 0), out_axes=0)(
            llm_state, llm_token
        )
    )
```
