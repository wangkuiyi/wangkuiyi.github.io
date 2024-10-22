# The Jacobian in JVP and VJP

<center>[Yi Wang](https://wangkuiyi.github.io/)</center>

This article presents my perspective on Jingnan Shi’s excellent post [Automatic Differentiation: Forward and Reverse](https://jingnanshi.com/blog/autodiff.html) from the viewpoint of a deep learning toolkit developer. (I worked on Paddle and PyTorch.) I also shamelessly import his CSS files.

Another interesting read is [The Autodiff Cookbook of JAX](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).  Although it doesn’t delve into the mathematical details, it references textbooks. If you find the math in those textbooks confusing, feel free to use this article as a supplement.

## Foundation of Deep Learning

The terms JVP (Jacobian-vector product) and VJP (vector-Jacobian product) are prevalent in the codebase and documentation of JAX and MLX. If you’re contributing to MLX, you may need to implement or override methods named jvp and vjp. These foundational concepts enable JAX and MLX to compute higher-order derivatives.

## JVP and VJP

JVP refers to the product of a Jacobian matrix $J$ and a vector $v$:

$$J\cdot v$$

Similarly, VJP refers to:

$$v\cdot J$$

**But what exactly is a Jacobian matrix, and why do we need these products?**

## $J$: The Jacobian Matrix

Consider a function $f$ that takes $n$ inputs, $x_1,\ldots,x_n$, and returns $m$ outputs, $y_1,\ldots,y_m$. We are interested in determining how much the $j$-th output changes when introducing a small change to the $i$-th input. This change is denoted as $\partial y_j / \partial x_i$.

The Jacobian matrix is a collection of these partial derivatives:

$$
J = 
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \ldots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\ 
\frac{\partial y_m}{\partial x_1} & \ldots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}
$$

**Why is each column of $J$ associated with $x_i$ and each row with $y_j$?**

## The V in JVP

The Jacobian matrix tells us how changes in $x_i$ affect $y_j$. For instance, if we change $x_1$ by $1$ while keeping the other $x_i$ values constant, the change in the $y_j$ values is:

$$
J
\cdot
\begin{bmatrix}
1 \\
0 \\
\vdots
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1}  \\
\vdots \\
\frac{\partial y_m}{\partial x_1} 
\end{bmatrix}
$$

This result makes sense becasue each $\frac{\partial y_j}{\partial x_1}$ is, by its definition, the change to $y_j$ due to the change to $x_1$ by 1.

In general, if each $x_i$ changes by $\epsilon_i$, represented as $v=[\epsilon_1,\ldots,\epsilon_n]^T$, the corresponding change to $y_j$ is:

$$
J 
\cdot 
\begin{bmatrix}
\epsilon_1 \\
\vdots \\
\epsilon_n
\end{bmatrix}
$$

**To make JVP the above sense, the Jacobian’s columns must correspond to $x_i$.**

## The V in VJP

In some situations, such as deep learning, the outputs $y_j$ might be passed into another function, say $g(y_1,\ldots,y_m)$, which could be a loss function that returns a scalar (the loss). The derivative of $g$, $\partial g/\partial y_j$, tells us how much the loss changes with respect to $y_j$. To understand how changes in $x_i$ affect the loss, we define:

$$
v = [\frac{\partial{g}}{\partial{y_1}}, \ldots, \frac{\partial{g}}{\partial{y_m}}]
$$

and compute VJP:

$$
\begin{aligned}
v \cdot J
&=
\left[
\sum_j \frac{\partial g}{\partial y_j} \frac{\partial y_j}{\partial x_1}, \ldots, \sum_j \frac{\partial g}{\partial y_j} \frac{\partial y_j}{\partial x_n}
\right]
\\
&=
\left[
\frac{\partial g}{\partial x_1}, \ldots, \frac{\partial g}{\partial x_n}
\right]
\end{aligned}
$$

**For the VJP to function correctly, each row of the Jacobian must correspond to $y_j$.**

## Derivatives Are Functions

Before diving into how JVP and VJP relate to forward-mode and reverse-mode autodiff, it’s essential to revisit a fundamental mathematical concept.

In the above discussion, we used $\frac{\partial y_j}{\partial x_i}$, which simplifies $\frac{\partial y_j(x_i)}{\partial x_i}$. The parentheses are crucial -- they indicate that $y_j$ is not a fixed value but a function depending on $x_i$.

Consider the simplest case where $n=m=1$, the function $f(x)$ takes a scalar value input and returns a scalar output.  Suppose that $f(x)=x^2$. The derivative of $f(x)$, denoted as $f'(x)$, is a function depending on $x$, just like $f(x)$ does.

$$
f'(x) = \frac{\partial f(x)}{\partial x} = 2x
$$

## Jacobians Are Functions

More generally, the Jacobian matrix consists of functions, not fixed values. It can also be seen as a function of $x={x_i,\ldots,x_n}$, returning a matrix of values:

$$
J(x)=
\begin{bmatrix}
\frac{\partial y_1(x_1)}{\partial x_1} & \ldots & \frac{\partial y_1(x_n)}{\partial x_n} \\
\vdots & \ddots & \vdots \\ 
\frac{\partial y_m(x_1)}{\partial x_1} & \ldots & \frac{\partial y_m(x_n)}{\partial x_n}
\end{bmatrix}
$$

### Example 1. Jacobian of $W x$

Consider the function $f$, which takes a vector $x$ and multiplies it by a matrix $W$:

$$
f(x)=W\cdot x
$$

where

$$
x=\begin{bmatrix}x_1\\ x_2 \end{bmatrix}
\qquad
W = \begin{bmatrix}w_{1,1} & w_{1,2} \\ w_{2,1} & w_{2,2} \end{bmatrix}
$$

Expanding the matrix multiplication, we have:

$$
y_1(x_1, x_2) = w_{1,1} x_1 + w_{1,2} x_2
$$
$$
y_2(x_1, x_2) = w_{2,1} x_1 + w_{2,2} x_2
$$

Thus, the Jacobian matrix is:

$$
J(x_1, x_2)
=
\begin{bmatrix}
\frac{\partial y_1(x_1,x_2)}{\partial x_1}, & \frac{\partial y_1(x_1,x_2)}{\partial x_2} \\
\frac{\partial y_2(x_1,x_2)}{\partial x_1}, & \frac{\partial y_2(x_1,x_2)}{\partial x_2} \\
\end{bmatrix}
=
\begin{bmatrix}w_{1,1} & w_{1,2} \\ w_{2,1} & w_{2,2} \end{bmatrix}
$$

This Jacobian matrix happens to consist of constants.

### Example 2. Jacobian of $x^T W x$

Now consider another function:

$$
f = x^T W x
$$

Expanding the matrix multiplication, we have:

$$
\begin{aligned}
f(x_1, x_2) 
&= 
(w_{1,1} x_1 + w_{2,1} x_2) x_1 + (w_{1,2} x_1 + w_{2,2} x_2) x_2
\\
&=
w_{1,1} x_1^2 + (w_{1,2}+w_{2,1})x_1 x_2 + w_{2,2} x_2^2
\end{aligned}
$$

The Jacobian matrix is:

$$
\begin{aligned}
J(x_1, x_2) 
&=
\begin{bmatrix}
\frac{\partial f(x_1, x_2)}{\partial x_1} &
\frac{\partial f(x_1, x_2)}{\partial x_2}
\end{bmatrix}
\\
&=
\begin{bmatrix}
2 w_{1,1} x_1 + (w_{1,2}+w_{2,1}) x_2 &
2 w_{2,2} x_2 + (w_{1,2}+w_{2,1}) x_1
\end{bmatrix}
\end{aligned}
$$

This Jacobian consists of functions depending on $x_1$ and $x_2$.

## JVP and VJP Are Functions

Since the Jacobian matrix is a function of $x$, both JVP and VJP are also functions:

$$\text{jvp}_f(x, v) = J_f(x)\cdot v$$

$$\text{vjp}_f(x, v) = v\cdot J_f(x)$$

Here, $J_f(x)$ is the matrix of partial derivative functions of $f$.

## Show Me the Code

Given a function $f(x_1,\ldots,x_n)$ returns a scalar, `jax.grad` computes its Jacobian:

$$
\begin{aligned}
J(x_1,\ldots,x_n) 
&= f'(x_1,\ldots,x_n)  \\
&= \left[ \frac{\partial f(x_1,\ldots,x_n)}{\partial x_1}, \ldots, \frac{\partial f(x_1,\ldots,x_n)}{\partial x_n} \right]
\end{aligned}
$$

Since $f(x_1,\ldots,x_n)$ returns a scalar, the Jacobian returned by `jax.grad(f)` is a row vector. Here's an example:

```python
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev

a = jnp.array([1, 1], dtype=jnp.float32)
b = jnp.array([1, 2], dtype=jnp.float32)
c = jnp.array([2, 1], dtype=jnp.float32)
d = jnp.array([2, 2], dtype=jnp.float32)

def f(x: jax.Array) -> jax.Array:
    """ f(x_1, x_2) = x_1 * x_2 """
    return x.prod()

print(grad(f)(a)) # [1. 1.]
print(grad(f)(b)) # [2. 1.]
print(grad(f)(c)) # [1. 2.]
print(grad(f)(d)) # [2. 2.]
```

Note that `jax.grad` only handle scalar-output functions.  For instance, calling `jax.grad` on a function returning a vector will raise an error:

```python
def f(x: jax.Array) -> jax.Array:
    "" f(x_1, x_2) = [ 11 x_1 + 33 x_2, 22 x_1 + 44 x_2 ] """
    w = jnp.array([[11, 22], [33, 44]], dtype=jnp.float32)
    return w @ x

print(jax.grad(f)(x))  # TypeError: Gradient only defined for scalar-output functions.
```

In this case, we should use `jax.jacfwd` or `jax.jacrev` instead of `jax.grad`.  `jax.jacfwd` uses idea behind JVP and `jax.jacrev` uses VJP, but both return very close if not identical results.  The following code works:

```python
print(jacfwd(f)(a))
print(jacfwd(f)(b))
print(jacrev(f)(c))
print(jacrev(f)(d))
```

All of these `print` calls display the same output because, as shown by Example 1, the Jacobian of `f` is a constant matrix:

```python
[[11. 22.]
 [33. 44.]]
```

The following code is for Example 2:

```python
def f(x: jax.Array) -> jax.Array:
    w = jnp.array([[11, 22], [33, 44]], dtype=jnp.float32)
    return x @ w @ x

print(jacfwd(f)(a))
print(jacfwd(f)(b))
print(jacfwd(f)(c))
print(jacfwd(f)(d))
```

The JAX funtion `jvp` takes a function `f`, the value `x` for calculating $f'(x)$, and `v` for calculating $f'(x)\cdot v$.

The JAX function `vjp` has a slightly different signature. It takes the function `f` and `x`, and returns another function that takes `v` and returns $f'(x)(v)$.
