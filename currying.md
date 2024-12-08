# Curried Functions and JAX Type Annotation

JAX's documentation uses type annotation conventions from type theory. Consider this example:

```python
lax.fori_loop(start: int, end: int, body: (int -> C -> C), init: C) -> C
```

The notation `body: (int -> C -> C)` indicates that `body` is a function taking two parameters: one of type `int` and another of type `C`, and returns a value of type `C`.  You might wonder why the first arrow `->` separates parameters while the second appears to annotate the return type.  This relates to the concept of [currying](https://en.wikipedia.org/wiki/Currying).

To understand currying, let's look at a simple example:

```python
def regular_add(x: int, y: float) -> str:
    return f"{x + y}"

def curried_add(x):
    def inner(y):
        return f"{x + y}"
    return inner

assert regular_add(1, 2) == curried_add(1)(2)
```

Here, `regular_add` takes two parameters (`int` and `float`) and returns a `str`.  In contrast, `curried_add` takes one parameter (`int`) and returns a function that takes a `float` and returns a `str`.  Though structured differently, these functions are functionally equivalent.  Therefore, the type annotation `int -> (float -> str)` could appropriately describe both functions.
