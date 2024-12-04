# JAX's vmap

[JAX documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) provides an example that sets both parameters, `in_axes` and `out_axes`, to the same value to explain `vmap`. However, it would be beneficial for users to have more examples with different values of `in_axes` and `out_axes` to better understand `vmap`.

## `in_axes`

The name `vmap` is derived from the well-known `map` function.  The innocation `map(f, l)` applies the function `f` to each element in the list `l`.  In contrast, the innocation `vmap(f)(v)` requires elements to be packed into a tensor `v`.

Consider that `f` processes a vector. We could pack the list of vectors as rows or columns of `v`. If they are rows, we should call `vmap(f, in_axies=0)(v)`, where `in_axies=0` indicates that the first dimension of `v` corresponds to the list. Alternatively, we call `vmap(f, in_axies=1)(v)` to specify that the columns of `v` are the elements, as illustrated in the following example:

```python
import jax
import jax.numpy as jnp

def f(x):
    return x.sum() # reduce a vector x

v = jnp.ones((3, 2))
print(f"v=\n{v}\n")
print(jax.vmap(f, in_axes=0)(v)) # batch=3
print(jax.vmap(f, in_axes=1)(v)) # batch=2
```

## `out_axes`

JAX is pure-functional, which means its tensors are immutable.  Therefore, when you call `jax.vmap(f)(v)`, it returns a new tensor.  The `out_axes` parameter instructs the function `jax.map(f)` on how to construct the resulting tensor.  If `out_axes=0`, `jax.map(f)` packs the outputs from each call to `f` as rows in the result tensor.  Alternatively, if `out_axes=1`, it packs them as columns, as illustrated in the following example:

```python
def g(x):
    return x

print(jax.vmap(g, in_axes=0, out_axes=0)(v))
print(jax.vmap(g, in_axes=0, out_axes=1)(v))
```
