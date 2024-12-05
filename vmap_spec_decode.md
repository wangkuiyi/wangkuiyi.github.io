# JAX's vmap: An Application of Speculative Decoding

In our recent work on a novel speculative decoding approach designed for Apple Intelligence, we prototyped the idea using [PyTorch](https://github.com/apple/ml-recurrent-drafter).

As other speculative decoding approaches, our approach involves sampling token sequences from a draft model.  When we implement this sampling algorithm in PyTorch, we write vectorized code.  This process can be a bit challenging.  We need to keep in mind that variables with a shape `[b, w, l, ...]`, where `b` is the batch size, each element in the batch corresponds to a prompt, `w` is the number of token sequences sampled for each prompt, and `l` is the length of each token sequence.  Additionally, we must select the appropriate operators to compose the vectorized algorithm.  The resulting code is less straightforward to understand for the rest of the team compared to the non-vectorized version.

This inspires me to attempt writing a non-vectorized and comprehensible version of the code. Then, I'll reply on JAX's vmap to automatically add the `b` and `w` dimensions.  The trial was successful. However, I also discovered that I need to define *point-free* functions, which are pure functions where **all local variables are exposed as parameters**.

## The RNN Drafter

Our approach employs an RNN as the draft model mandated by speculative decoding. Similar to most RNNs, its forward pass takes an input token and predicts the subsequent token.  This process also utilizes and modifies the RNN state, which is represented as a vector of length `h`.

When we repeatedly invoke the RNN for `l` iterations, it should generate `l` tokens.  The initial input token is sampled from the target LLM because, in speculative decoding, we use the draft model to predict what the LLM will generate.  Additionally, using the RNN for this purpose necessitates the inclusion of an extra parameter: the hidden state of the token sampled from the target LLM.  A simplified version of our RNN draft model is as follows:

```python
def rnn_drafter(llm_state, rnn_state, input_token) -> Tuple[jax.Array, jax.Array]:
    """[d], [d'], [] -> [d'], []"""
    print(f"\ninput_token=\n{input_token}")
    print(f"\nrnn_state=\n{rnn_state}")
    return rnn_state * 2, input_token + 1
```


