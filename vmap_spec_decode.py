import jax
import jax.numpy as jnp
from typing import Tuple

d = 8  # hidden dim of llm
d1 = 4  # hidden dim of rnn
l = 2  # number of tokens drawn from rnn after the one from llm
w = 3  # beam width
b = 2  # batch size


def rnn_drafter(llm_state, rnn_state, input_token) -> Tuple[jax.Array, jax.Array]:
    """[d], [d'], [] -> [d'], []"""
    print(f"\ninput_token=\n{input_token}")
    print(f"\nrnn_state=\n{rnn_state}")
    return rnn_state * 2, input_token + 1


def sample_seq(llm_state, llm_token) -> jax.Array:
    """[d'], [] -> [l]"""
    token = llm_token
    rnn_state = jnp.zeros((d1,))
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


def sample_seq_no_local_var(llm_state, rnn_state, llm_token) -> jax.Array:
    """[d'], [d], [] -> [l]"""
    token = llm_token
    seq = jnp.array([token], dtype=jnp.int32)
    for _ in range(l):
        rnn_state, token = rnn_drafter(llm_state, rnn_state, token)
        seq = jnp.append(seq, token)
    return seq


def test_sample_beam_no_local_var():
    llm_state, rnn_state, llm_token = (
        jnp.zeros((d,)),
        jnp.zeros((w, d1)),
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
        jnp.zeros((b, w, d1)),
    )
    llm_token = jnp.tile(jnp.arange(0, w * 100, 100), (b, 1))
    print(
        jax.vmap(
            jax.vmap(sample_seq_no_local_var, in_axes=(None, 0, 0), out_axes=0),
            in_axes=(0, 0, 0),
            out_axes=0,
        )(llm_state, rnn_state, llm_token)
    )
