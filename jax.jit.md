# Decipher JAX's Tracing and JIT Compilation

JAX provides JIT compilation through the `jax.jit` API, as shown in the following example:

```python
fsin = jax.jit(jnp.sin)
x = jnp.linspace(0, 1.0, 5)
print(jnp.array_equal(jnp.sin(x), fsin(x)))
```

This example may lead users to believe that `jax.jit(jnp.sin)` compiles and returns a “faster” version of `jnp.sin`.  However, in reality, the first call to `fsin` triggers the actual compilation.

This misconception can lead to further confusion about the code’s behavior. For example, users might assume that jax.jit(jnp.sin) is time-consuming due to compilation. However, it is the call to `fsin(x)` that initiates the compilation and thus takes significant time.

More importantly, this misconception may prevent users from understanding JAX’s requirement for fixed-shape input arrays during compilation.  The call `jax.jit(jnp.sin)` alone does not involve any input arrays, which is why the actual compilation happens only when `fsin` is called with an input array.

## Compile and Cache

The following example demonstrates that the initial call to a function decorated with `jax.jit` triggers the time-consuming compilation process, while subsequent calls execute much faster due to caching.

```python
import jax.numpy as jnp

x = jnp.linspace(0., 1., 1000)
%time jax.jit(jnp.sin)(x).block_until_ready()  # 33.6 ms
%time jax.jit(jnp.sin)(x).block_until_ready()  # 852 µs
%time jax.jit(jnp.sin)(x).block_until_ready()  # 910 µs
%time jax.jit(jnp.sin)(x).block_until_ready()  # 891 µs
```

## Timing Asynchronous Operations

In the example above, calls to [`block_until_ready`](https://github.com/jax-ml/jax/blob/7bc026e496ebe49d54d6578746dd9c9beaed2592/jax/_src/array.py#L597) ensure that the results are fully computed.  According to the JAX documentation on [Asynchronous Dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html), `jax.Array` is a future -- a placeholder for a value that will be computed on an accelerator device but may not be available immediately.  Calling `block_until_ready` forces the program to wait for the execution of `jax.jit(jnp.sin)` to complete and return the result.

Asynchronous dispatch is useful because it enables Python programs to enqueue substantial amounts of work for the accelerator.  MLX adopts a similar design. To ensure an array `x` is ready in MLX, you can call `mx.eval(x)`.

## Tracing with `jax.make_jaxpr`

[JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables) mentions that JIT uses `jax.make_jaxpr` to "trace" Python code and produce an intermediate representation called JAXPR.  However, it does not reveal details about `make_jaxpr`.  So I crafted the following example allows a peek into the hole.

```python
import jax
import jax.numpy as jnp

def f(x, y) -> jax.Array:
    print("type(x):", type(x))
    print("x:", x)
    print("type(y):", type(y))
    print("y:", y)
    return jnp.dot(x + 1, y + 1)
```

The normal way to call this function is to pass in two arrays:

```python
x = jnp.array([1.0, 2.0])
y = jnp.array([1.0, 2.0])
print(f(x, y))
```

The function `f` prints the type and value of `x` and `y`, as well as the final result `13`.

```text
type(x): <class 'jaxlib.xla_extension.ArrayImpl'>
x: [1. 2.]
type(y): <class 'jaxlib.xla_extension.ArrayImpl'>
y: [1. 2.]
13.0
```

Now let us check what `jax.make_jaxpr` returns:

```python
ff = jax.make_jaxpr(f)
print("type(ff):", type(ff))
print("ff:", ff)
```

It returns a function:

```text
type(ff): <class 'function'>
ff: <function make_jaxpr(f) at 0x10a06af80>
```

Let us try calling the function:

```python
z = ff(x, y)
print("type(z):", type(z))
print("z:", z)
```

This prints the type and value of `x` and `y` as well as the returned value:

```text
type(x): <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>
x: Traced<ShapedArray(float32[2])>with<DynamicJaxprTrace(level=1/0)>
type(y): <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>
y: Traced<ShapedArray(float32[2])>with<DynamicJaxprTrace(level=1/0)>
type(z): <class 'jax._src.core.ClosedJaxpr'>
z: { lambda ; a:f32[2] b:f32[2]. let
    c:f32[2] = add a 1.0
    d:f32[2] = add b 1.0
    e:f32[] = dot_general[
      dimension_numbers=(([0], [0]), ([], []))
      preferred_element_type=float32
    ] c d
  in (e,) }
```

We passed in two arrays to `ff`.  However, the calls to `print` by `f` show that `x` and `y` are of type `DynamicJaxprTracer`, not arrays.  Obviously, the function `ff`, created by `jax.make_jaxpr`, calls `f`, which is why the `print` calls in `f` work.  But, before calling `f`, `ff` converts the input arrays into `DynamicJaxprTracer`.

The `DynamicJaxprTracer` contains only the `ShapedArray` with `float32` dtype and shape `[2]`; the actual data is missing.  That is the purpose of tracing: capturing the dtype and shape of arrays but not the content.

As expected, the return value is not an array but a representation of the operations within `f`.  For this exmaple, it is a short program that calls the [XLA operation `dot_general`](https://openxla.org/xla/operation_semantics#dotgeneral).

From this, we can infer how `jax.make_jaxpr` works.  It returns a function that calls `f` with arguments converted to `DynamicJaxprTracer` instances, capturing dtype and shape while allowing functions like jnp.dot to treat them like arrays.  Thanks to Python's support of [duck-typing](https://en.wikipedia.org/wiki/Duck_typing), JAX functions can operate on tracers like they are operating arrays.

```python
def make_jaxpr(f):
    def ff(*args, **kwargs):
        args, kwargs = convert_arrays_to_tracer(args, kwargs)
        return f(args, kwargs)
    return ff
```

## Compile the Trace

Now, let us hypothesize how `jax.jit` work.  According to the initial example, `jax.jit` takes a function `f` as input.  If `f` has already been compiled, `jax.jit` should return the cached result.  If not, `jax.jit` should return a function `ff` that runs the identical operations as `f`.  When called with arguments like `x` and `y`, `ff` would:

1. trace `f` given `x` and `y`,
1. call XLA to compile the tracing result,
1. cache the compiled result, and
1. calls the result with `x` and `y`.

The source code of `jax.jit` may look something like the following:

```python
def jit(f):
   if ff := cached.of(f):
       return ff
       
   def trigger(*args, **kwargs):
       ff = jax.make_jaxpr(f)
       trace = f(args, kwargs)
       compiled = compile_using_xla(trace)
       cached.add(compiled)
       return compiled(args, kwargs)
   return tigger
```

## Conclusion

I haven't read the JAX codebase to verify if my hypothesis is correct.  But I plan to. :-)
