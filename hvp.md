# Evaluate Hessian-Vector Product Without The Hessian Matrix

<center>[Yi Wang](https://wangkuiyi.github.io/)</center>


The [JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-with-grad-of-grad) provides a concise one-liner to compute the Hessian-vector product (HVP) of a scalar-valued function $f(x)$.  This method is highly efficient and avoids the need to store the potentially large Hessian matrix.

```python
def hvp(f, x, v):
    return jax.grad(lambda x: jax.vdot(jax.grad(f)(x), v))(x)
```

The mathematical explanation is succinct and uses higher-order function annotations:

$$
\partial^2 f(x) \cdot v =
\partial[x \mapsto \partial f(x) \cdot v]
$$

Here is its representation in standard calculus notation:

$$
\frac{\partial^2 f(x)}{\partial x} \cdot v
=
\frac{\partial \left( \frac{\partial f(x)}{\partial x} \cdot v \right)}{\partial x}
$$

In this equation, $\frac{\partial f(x)}{\partial x}$ is the [Jacobian](jacobian.html) and $\frac{\partial f(x)}{\partial x} \cdot v$ is the JVP function that depends on $x$ and $v$.

The equation states that the derivative of the JVP with respect to $x$ is the HVP.  To see why this holds, we start with the definitions of the Jacobian and Hessian for a scalar-valued function $f(x)$.

## HVP By Definition

Since $f(x)$ is scalar-valued, its Jacobian is a function that returns a row vector over $x=\{x_1,\ldots,x_n\}$.  An equivalent intepretation of the Jacobian is a row vector of partial derivative functions:

$$
J_f(x) =
\begin{bmatrix}
\frac{\partial f(x)}{\partial x_1}, \ldots, \frac{\partial f(x)}{\partial x_n}
\end{bmatrix}
$$

The Hessian is the derivative, also a function, of the Jacobian with respect to each variable in $\{x_1,\ldots,x_n\}$.  Similar to the Jacobian, the Hessian could be interpeted as a function that returns a matrix of values, or equivalently, a matrix of functions, each of which returns a single value.

$$
H_f(x) =
\begin{bmatrix}
\frac{\partial^2 f(x)}{\partial x_1 \partial x_1}, & \ldots, & \frac{\partial^2 f(x)}{\partial x_n \partial x_1}  \\
\vdots                              & \ddots, & \vdots \\
\frac{\partial^2 f(x)}{\partial x_1 \partial x_n}, & \ldots, & \frac{\partial^2 f(x)}{\partial x_n \partial x_n}  \\
\end{bmatrix}
$$

The HVP of $f(x)$ is a function that takes two vectors, $x$ and $v$, and returns a matrix:

$$
\text{HVP}_f(x)
=
H_f(x) \cdot v
=
\begin{bmatrix}
\frac{\partial^2 f(x)}{\partial x_1 \partial x_1}, & \ldots, & \frac{\partial^2 f(x)}{\partial x_n \partial x_1}  \\
\vdots                              & \ddots, & \vdots \\
\frac{\partial^2 f(x)}{\partial x_1 \partial x_n}, & \ldots, & \frac{\partial^2 f(x)}{\partial x_n \partial x_n}  \\
\end{bmatrix}
\cdot
\begin{bmatrix}
v_1 \\
\vdots \\
v_n
\end{bmatrix}
=
\begin{bmatrix}
\sum_i \frac{\partial^2 f(x)}{\partial x_i \partial x_1} v_i \\
\vdots \\
\sum_i \frac{\partial^2 f(x)}{\partial x_i \partial x_n} v_i 
\end{bmatrix}
$$

## HVP as Derivative of JVP

The JVP function of $f(x)$ returns a vector $J_f(x)\cdot v$:

$$
\text{JVP}_f(x, v) 
= J_f(x)\cdot v 
=
\sum_i \frac{\partial f(x)}{\partial x_i} v_i
$$

The derivative of $\text{JVP}_f(x,v)$ with respect to $x$ is:

$$
\begin{bmatrix}
\frac{\partial \sum_i \frac{\partial f(x)}{\partial x_i} v_i}{\partial x_1} \\
\vdots \\
\frac{\partial \sum_i \frac{\partial f(x)}{\partial x_i} v_i}{\partial x_n} \\
\end{bmatrix}
=
\begin{bmatrix}
\sum_i  \frac{\partial^2 f(x)}{\partial x_i \partial x_1} v_i \\
\vdots \\
\sum_i  \frac{\partial^2 f(x)}{\partial x_i \partial x_n} v_i \\
\end{bmatrix}
$$

This result aligns precisely with the definition of the HVP.

## An Example

JAX provides the function `jax.jvp`, a generalized form of `jnp.vdot(jax.grad(f), v)` capable of handling input functions that returns multiple values.  The following example demonstrates how to define `hvp` using `jax.jvp`.

```python
import jax
import jax.numpy as jnp

def hvp(f, x, v):
    return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)

def f(x):
    return jnp.sin(x).sum()

x, v = jnp.array([0.0, 1.0]), jnp.array([10.0, 10.0])

print(hvp(f, x, v))

primal, tangent = jax.jvp(f, (x,), (v,))
assert tangent == jnp.vdot(jax.grad(f)(x), v)

def hvp(f, x, v):
    return jax.grad(lambda x: jax.jvp(f, (x,), (v,))[1])(x)

print(hvp(f, jnp.array([0.0, 1.0]), jnp.array([10.0, 10.0])))
```
