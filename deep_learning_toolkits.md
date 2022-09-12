# About Deep Learning Toolkits

My recent work on PyTorch Distributed and TorchRec requires me to learn PyTorch 2.0.  At the same time, I am learning JAX and XLA from Alpa authors in my spare time.  Looking back from these technologies in 2022 at older generations of technologies, it seems that various deep learning toolkits are trying to address the two critical challenges:

1. functional transformations, including autograd and parallelizations such as vmap, pmap, and pjit, and,
2. heterogeneous computing, the CPU takes care of the control flow and the GPU/TPU takes care of tensor computation and collective communication.

## Functional Transformation

I use the term "functional transformation" here to mean changing one procedure into another. The most common example is autograd, which takes the forward procedure written by users and creates the backward procedure, which is usually too complex for users to write. Functional transformation raises the question of how to represent the input and output procedures so that it is easy to write the functional transformation algorithm.

### Theano: Explicitly Build the IR

Theano, now known as the Apsara project, was one of the first deep learning toolkits. It has an API that lets users build the IR as a data structure in memory. Then, we can tell Theano to do the autograd and turn the result into a Python function.

```python
import aesara
from aesara import tensor as at

a = at.dscalar("a") # Define placeholders, which have no values.
b = at.dscalar("b")

c = a * b              # c now contains the IR of an expression.TT
dc = aesara.grad(c, a) # Convert the IR in c into another one, dc

f_dc = aesara.function([a, b], dc) # Convert the IR into a Python function,
assert f_dc(1.5, 2.5) == 2.5       # so we can call it.
```

You can run this [Colab notebook](https://colab.research.google.com/drive/1eg7C5WMNokhXgXQ46pNA30dXUCklquPz?usp=sharing) that includes the above code snippet.

### TensorFlow 1.x: A VM to Run the IR

TensorFlow 1.x keeps the idea of building the IR explicitly. In TensorFlow, the above example looks almost the same. With TensorFlow 1.x, the main difference is that we don't turn the backward IR into a Python function and then use the Python interpreter to run it. Instead, we send the IR to the TensorFlow runtime service to run it.

```
import tensorflow.compat.v1 as tf # TensorFlow 1.x API
import numpy as np
tf.disable_eager_execution()

a = tf.placeholder(tf.float32, shape=(2, 2))
b = tf.placeholder(tf.float32, shape=(2, 2))

c = tf.matmul(a, b)
dc = tf.gradients(c, [a], stop_gradients=[a, b])

with tf.compat.v1.Session() as sess:  # TensorFlow has a runtime to execute the IR,
  x = np.random.rand(2, 2)   `       `  # so, no converting it into Python code.
  print(sess.run(dc, feed_dict={a:x, b:x}))
```

You can run this [Colab notebook](https://colab.research.google.com/drive/1jc0ePg2AAXBihevtoZM_33mmhC70rzqz?usp=sharing) that includes the above code snippet.

### PyTorch 1.x: No IR for Forward

PyTorch does not turn the forward pass into an IR like Theano or TensorFlow does. Instead, it uses the Python interpreter to run the forward pass. During this run, an IR representing the backward pass is built as a side effect. This is known as the "eager mode".

```
import torch

a = torch.tensor(1.0, requires_grad=True) # These are not placeholders, but values.
b = torch.tensor(2.0)

c = a * b    # Evaluates c and derives the IR of the backward in c.grad_fn_.
c.backward() # Executes c.grad_fn_.
print(c.grad)
```

You can run this [Colab notebook](https://colab.research.google.com/drive/1v4hENL-IJ-C6VT5H9W1NC2te85D8VdJK?usp=sharing) that includes the above code snippet.

### TensorFlow 2.x: Gradient Tape

TensorFlow 2.x adds an eager mode API like PyTorch's. This API traces how the forward pass was run into an IR called the GradientTape. TensorFlow 2.x can figure out the backward pass from this trace.

```
import tensorflow as tf

a = tf.Variable(1.0) # Like PyTorch, these are values, not placehodlers.
b = tf.Variable(2.0)

with tf.GradientTape() as tape:
  c = a * b
dcda = tape.gradient(c, a)
print(dcda)
```

You can run this [Colab notebook](https://colab.research.google.com/drive/1PbftzJ9E2_FyIiuozTpExMvlFky_G2nv?usp=sharing) that includes the above code snippet.

### JAX

JAX does not expose low-level details like GradientTape to users. The JAX way of thinking, on the other hand, is that both the input and output functions are just Python functions.

```
import jax 

a = 2.0
b = 3.0
jax.grad(jax.lax.mul)(a, b)  # Compute c = a * b w.r.t. a.  The result is b=3.
```

For advanced users who want to write their own functional transformations, they can call low-level APIs like `make_jaxpr` to get access to the IR, which is known as JAXPR.

```
jax.make_jaxpr(jax.lax.mul)(2.0, 3.0)  # Returns the IR representing jax.lax.mul(2,3)
jax.make_jaxpr(jax.grad(jax.lax.mul))(2.0, 3.0)  # Returns the IR of grad(mul)(2,3)
```

You can run this [Colab notebook](https://colab.research.google.com/drive/1PlFijLIzAttIBd3tBjiEbSgPXvq9lVlg?usp=sharing) that includes the above code snippets.

### FuncTorch

[functorch](https://github.com/pytorch/functorch) is a JAX-like function transformation based on PyTorch.

```
import torch, functorch

a = torch.tensor([2.0])
b = torch.tensor([3.0])
functorch.grad(torch.dot)(a, b)
```

JAX’s `make_jaxpr` is analogous to `make_fx` from functorch.

```
def f(a, b):
  return torch.dot(a, b) # Have to wrap the builtin function dot into f.
  
print(functorch.make_fx(f)(a, b).code)
print(functorch.make_fx(functorch.grad(f))(a, b).code)
```

TensorFlow 2.x, JAX, and functorch all build an IR for the forward pass, but PyTorch eager mode does not. Not only is the IR useful for autograd, but it is also useful for other kinds of functional transformations. In the following example, `functorch.compile.aot_function` will invoke the callback `print_compile_fn` twice, once for the forward pass and once for the backward pass.

```
from functorch.compile import aot_function
import torch.fx as fx

def print_compile_fn(fx_module, args):
    print(fx_module)
    return fx_module
aot_fn = aot_function(torch.dot, print_compile_fn)
aot_fn(a, b)
```

You can run this [Colab notebook](https://colab.research.google.com/drive/1o-yJ-5g1V084RDaiRw2PqfAjOG7Ty951?usp=sharing) that includes the above code snippets.

## Heterogeneous Computing



### Dynamic Control Flows

