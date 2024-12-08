from typing import Callable, Tuple

import jax
import jax.numpy as jnp


def beam_search(
    logits_fn: Callable,
    init_tokens: jax.Array,
    init_scores: jax.Array,
    max_len: int,
):
    """
    logits_fn(beam [beam_width, max_len], cur_len: int, vocab_size: int):
        returns next-token logits in shape [beam_size, vocab_size].
    init_tokens: [beam_width]
    init_scores: [beam_width]
    Returns: sequences of shape (beam_width, max_len)
    """
    beam_width = init_tokens.shape[0]

    def step(cur_len: int, state):
        beam, scores = state

        next_logits = logits_fn(beam, cur_len)  # (beam_width, vocab_size)
        vocab_size = next_logits.shape[-1]

        next_scores = scores[:, None] + next_logits  # (beam_width, vocab_size)

        # Get top beam_width candidates
        flat_scores = next_scores.reshape(-1)
        top_indices = jax.lax.top_k(flat_scores, beam_width)[1]

        # Convert to sequence and token indices
        seq_idx = top_indices // vocab_size
        token_idx = top_indices % vocab_size

        # Update beam and scores must have fixed shape as required
        # by jax.lax.fori_loop
        new_beam = beam[seq_idx]
        new_beam = new_beam.at[:, cur_len].set(token_idx)
        new_scores = flat_scores[top_indices]

        return (new_beam, new_scores)

    init_beam = jnp.tile(
        init_tokens.reshape(-1, 1), (1, max_len)
    )  # (beam_width, max_len)
    final_state = jax.lax.fori_loop(1, max_len, step, (init_beam, init_scores))
    return final_state[0]


batched_beam_search = jax.vmap(beam_search, in_axes=(None, 0, 0, None), out_axes=0)


def make_logits_fn(vocab_size):
    def fn(beam, cur_len):
        # Always prefer the next token in the vocabulary.
        transition_logits = (
            jnp.eye(vocab_size, k=1) + jnp.eye(vocab_size, k=-(vocab_size - 1)) + 0.1
        )
        last_tokens = beam[:, cur_len - 1]
        return transition_logits[last_tokens]  # [beam_width, vocab_size]

    return fn


def test_beam_search():
    vocab_size = 5
    beam_width = 2
    max_len = 3
    init_tokens = jnp.array([0, 1], dtype=jnp.int32)
    init_scores = jnp.zeros(beam_width)

    beam = beam_search(make_logits_fn(vocab_size), init_tokens, init_scores, max_len)
    expected = jnp.array([[0, 1, 2], [1, 2, 3]])
    assert beam.shape == (beam_width, max_len)
    assert jnp.array_equal(beam, expected)


def test_batched_beam_search():
    batch_size = 2
    vocab_size = 5
    beam_width = 2
    max_len = 3
    init_tokens = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    init_scores = jnp.zeros((batch_size, beam_width))

    beam = batched_beam_search(
        make_logits_fn(vocab_size), init_tokens, init_scores, max_len
    )
    expected = jnp.array(
        [
            [[0, 1, 2], [1, 2, 3]],  # batch 0
            [[2, 3, 4], [3, 4, 0]],  # batch 1
        ]
    )
    assert beam.shape == (batch_size, beam_width, max_len)
    assert jnp.array_equal(beam, expected), f"Expected {expected}, got {beam}"
