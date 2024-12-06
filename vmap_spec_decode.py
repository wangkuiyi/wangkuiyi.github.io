"""

## Speculative Decoding

Large language models (LLMs) traditionally use auto-regressive decoding for text generation, which requires calling the LLM once for each token in the output sequence. For a sequence of length L, this means making L separate calls to the LLM. Speculative decoding offers a more efficient approach by reducing these costly LLM calls.

The key innovation of speculative decoding is the introduction of a smaller "draft" model that works alongside the LLM. This draft model, which is computationally inexpensive compared to the LLM, proposes l tokens at once. The LLM then verifies these tokens in a single call, accepting anywhere from 1 to l tokens (denoted as l', where 0 < l' ≤ l).

Since each successful speculation requires only one LLM call to verify multiple tokens, and the draft model's computational cost is negligible, speculative decoding achieves significant speedup over traditional auto-regressive decoding. The total number of LLM calls becomes substantially less than L, leading to faster overall inference times.

## Vectorzied Drafting

We had to keep in mind that variables with a shape `[b, w, l, ...]`, where `b` is the batch size, each element in the batch corresponds to a prompt, `w` is the number of token sequences sampled for each prompt, and `l` is the length of each token sequence.  Additionally, we must select the appropriate operators to compose the vectorized algorithm.  The resulting code is less straightforward to understand for the rest of the team compared to the non-vectorized version.


## The RNN Drafter

Our approach employs an RNN as the draft model mandated by speculative decoding. Similar to most RNNs, its forward pass takes an input token and predicts the subsequent token.  This process also utilizes and modifies the RNN state, which is represented as a vector of length `h`.

When we repeatedly invoke the RNN for `l` iterations, it should generate `l` tokens.  The initial input token is sampled from the target LLM because, in speculative decoding, we use the draft model to predict what the LLM will generate.  Additionally, using the RNN for this purpose necessitates the inclusion of an extra parameter: the hidden state of the token sampled from the target LLM.  A simplified version of our RNN draft model is as follows:

```python
def rnn_drafter(llm_state, rnn_state, input_token) -> Tuple[jax.Array, jax.Array]:
    """[d], [d'], [] -> [d'], []"""
    print(f"\nrnn_state=\n{rnn_state}")
    rnn_state = rnn_state + input_token
    return rnn_state, input_token + rnn_state.sum()
```

The above simplification is similar to our RNN model in that the update of `rnn_state` depends on `input_token`, and the output token depends on the updated `rnn_state`. Additionally, the summation of `rnn_state` and `input_token` implies that both arrays have the same dimensionality in their first dimensions.

We don’t need to worry about vectorization while writing the above code. We don’t have to keep in mind that later, when we sample tokens, we’ll have a beam of input tokens instead of a single token. We don’t have to worry about ensuring that each input token in the beam corresponds to a corresponding row in `rnn_state` or `llm_state`, or that `llm_state` isn’t vectorized with respect to the beam width.

## Sample a Sequence of Tokens

Given the above non-vectorized forward pass of our RNN draft model, it is trivial to write a loop to sample a sequence of tokens given an initial one.

```python
def sample_seq(llm_state, llm_token) -> jax.Array:
    """[h], [] -> [l]"""
    token = llm_token
    rnn_state = jnp.zeros((h,))
    seq = jnp.array([token], dtype=jnp.int32)
    for _ in range(l):
        rnn_state, token = rnn_drafter(llm_state, rnn_state, token)
        seq = jnp.append(seq, token)
    return seq
```

"""
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


def sample_seq_no_local_var(llm_state, rnn_state, llm_token) -> jax.Array:
    """[h], [d], [] -> [l]"""
    token = llm_token
    seq = jnp.array([token], dtype=jnp.int32)
    for _ in range(l):
        rnn_state, token = rnn_drafter(llm_state, rnn_state, token)
        seq = jnp.append(seq, token)
    return seq


def test_sample_beam_no_local_var():
    llm_state, rnn_state, llm_token = (
        jnp.zeros((d,)),
        jnp.zeros((w, h)),
        jnp.arange(0, w * 100, 100),
    )
    print(
        jax.vmap(sample_seq_no_local_var, in_axes=(None, 0, 0), out_axes=0)(
            llm_state, rnn_state, llm_token
        )
    )


def test_sample_beam_no_local_var_batched():
    llm_state, rnn_state = (
        jnp.zeros((b, d)),
        jnp.zeros((b, w, h)),
    )
    llm_token = jnp.tile(jnp.arange(0, w * 100, 100), (b, 1))
    print(
        jax.vmap(
            jax.vmap(sample_seq_no_local_var, in_axes=(None, 0, 0), out_axes=0),
            in_axes=(0, 0, 0),
            out_axes=0,
        )(llm_state, rnn_state, llm_token)
    )

