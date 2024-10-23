# Automatic Differentiation (Part 2): Built a Simple Deep Learning Toolkit with VJP

<center>[Yi Wang](https://wangkuiyi.github.io/)</center>

This article explains automatic differentiation for neural networks using JVP and provides a concise Python implementation. For a more detailed explanation of VJP and JVP, I recommend my post [*The Jacobian in JVP and VJP*](jacobian.html).

Alternative approaches to automatic differentiation include symbolic differentiation, as implemented in MATLAB and Mathematica, and numerical differentiation, which is straightforward to implement. Jingnan Shi’s article [*Automatic Differentiation: Forward and Reverse*](https://jingnanshi.com/blog/autodiff.html) offers a comparison between these methods and automatic differentiation.

## An Example

Let us develop a very simple deep learning toolkit. Consider a neural network that computes the following expression. It should be able to compute $\partial y/\partial x_1$ and $\partial y/\partial x_2$ for specific values of $x_1=\pi/2$ and $x_2=1$.

$$
y(x_1, x_2) = \sin(x_1) \cdot (x_1 + x_2)
$$

The following figure illustrates the forward and backward passes:

<center><img src="autodiff.svg" /></center>

## The Design

The above figure presents a computation graph. In C++, you might use pointers for the links in the graph. However, for easier memory management, we store the nodes in an array and use their indices as pointers. We can encapsulate this array in a class called `Tape`.

Except for the node $y = v_3$, which serves as an annotation, each of the other nodes represents an operation. Thus, we need a class `Op` to represent operations.

There are four different operations in this graph, so we will derive `Sine`, `Add`, `Mult`, and `Var` from `Op`, where `Var` represents the input variables.

The class `Op` must have a field `value`.

The constructor of the `Op` class can take inputs and compute the `value` field.

It is easier to initialize an `Op` object by calling a method of `Tape`, allowing the method to append the `Op` instance to the tape. As a result, we need methods like `Tape.var`, `Tape.sin`, `Tape.add`, and `Tape.mult`.

In this way, the last instance in the tape is the final output, or $v_3$ in the above example.

The backward pass starts by calling a method on the last `Op` instance. Since the last operation is the final output $y$, $\frac{\partial y}{\partial v_3}$ is 1. As explained in [*The Jacobian in JVP and VJP*](jacobian.html), $\frac{\partial y}{\partial v_3}$ is the *v* in VJP. Thus, we add the method `Op.vjp` to handle the backward pass.

The operation $v_3$ has two inputs. Its `vjp` method should either pass $\frac{\partial y}{\partial v_3}$ to each input operation's `vjp` method, allowing their `vjp` methods to compute $\frac{\partial y}{\partial v_3} \frac{\partial v_3}{\partial v_1}$ and $\frac{\partial y}{\partial v_3} \frac{\partial v_3}{\partial v_2}$, or it could compute the gradients for its inputs and store the results directly into its input operations. The latter approach is easier because the `Mult` operation knows how to compute its backward pass, i.e., $\frac{\partial v_3}{\partial v_1}$ and $\frac{\partial v_3}{\partial v_2}$ in this case.

It’s important to note that before computing and propagating the VJP result, `op.vjp` must wait for and accumulate gradients from all subsequent operations that use `op`'s value. For instance, the operation $x_1$ is used by $v_1$ and $v_2$, so it has to accumulate $\frac{\partial y}{\partial v_1} \frac{\partial v_1}{\partial x_1}$ from $v_1$`.vjp` and $\frac{\partial y}{\partial v_2} \frac{\partial v_2}{\partial x_1}$ from $v_2$`.vjp`.

To track how many gradients an operation needs to wait for, we add a field `Op.succ`. To track how many it has received, we introduce another field `Op.recv`.

## The Code

The following program, `autodiff.py`, implements the above design.  You can run it using `pytest autodiff.py`.

```python
import math
from typing import List


class Op:
    def __init__(self):
        self.value: float = 0.0
        self.grad: float = 0.0
        self.recv: int = 0
        self.succ: int = 0
        self.srcs: List[int] = []

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            + f"value:{self.value} grad:{self.grad} recv:{self.recv}-of-{self.succ} "
            + "srcs: "
            + "".join([f"{i} " for i in self.srcs])
        )


class Var(Op):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def vjp(self, tape: "Tape", grad: float):
        self.recv += 1
        self.grad += grad


class Sine(Op):
    def __init__(self, x: Op):
        super().__init__()
        self.value = math.sin(x.value)

    def vjp(self, tape: "Tape", grad: float):
        self.grad += grad
        self.recv += 1
        if self.recv >= self.succ:
            src = tape.tape[self.srcs[0]]
            src.vjp(tape, math.cos(src.value) * self.grad)


class Add(Op):
    def __init__(self, x: Op, y: Op):
        super().__init__()
        self.value = x.value + y.value

    def vjp(self, tape: "Tape", grad: float):
        self.grad += grad
        self.recv += 1
        if self.recv >= self.succ:
            x, y = self.srcs[0], self.srcs[1]
            tape.tape[x].vjp(tape, self.grad)
            tape.tape[y].vjp(tape, self.grad)


class Mult(Op):
    def __init__(self, x: Op, y: Op):
        super().__init__()
        self.value = x.value * y.value

    def vjp(self, tape: "Tape", grad: float):
        self.grad += grad
        self.recv += 1
        if self.recv >= self.succ:
            x, y = self.srcs[0], self.srcs[1]
            tape.tape[x].vjp(tape, tape.tape[y].value * self.grad)
            tape.tape[y].vjp(tape, tape.tape[x].value * self.grad)


class Tape:
    def __init__(self):
        self.tape: List[Op] = []

    def __str__(self) -> str:
        return "\n".join([str(op) for op in self.tape])

    def var(self, v: float) -> int:
        self.tape.append(Var(v))
        return len(self.tape) - 1

    def sin(self, x: int) -> int:
        self.tape.append(Sine(self.tape[x]))
        r = len(self.tape) - 1
        self.tape[x].succ += 1
        self.tape[r].srcs.append(x)
        return r

    def add(self, x: int, y: int) -> int:
        self.tape.append(Add(self.tape[x], self.tape[y]))
        r = len(self.tape) - 1
        self.tape[x].succ += 1
        self.tape[y].succ += 1
        self.tape[r].srcs.append(x)
        self.tape[r].srcs.append(y)
        return r

    def mult(self, x: int, y: int) -> int:
        self.tape.append(Mult(self.tape[x], self.tape[y]))
        r = len(self.tape) - 1
        self.tape[x].succ += 1
        self.tape[y].succ += 1
        self.tape[r].srcs.append(x)
        self.tape[r].srcs.append(y)
        return r

    def backprop(self, r: int):
        self.tape[r].vjp(self, 1.0)


def test_add():
    t = Tape()
    r = t.add(t.var(2), t.var(3))
    assert t.tape[r].value == 5
    t.backprop(r)
    assert t.tape[0].grad == 1
    assert t.tape[1].grad == 1


def test_mult():
    t = Tape()
    r = t.mult(t.var(2), t.var(3))
    assert t.tape[r].value == 6
    t.backprop(r)
    assert t.tape[0].grad == 3
    assert t.tape[1].grad == 2


def test_sin():
    t = Tape()
    r = t.sin(t.var(0.0))
    assert t.tape[r].value == 0
    t.backprop(r)
    assert t.tape[0].grad == 1.0


def test_compound():
    t = Tape()
    x1 = t.var(math.pi / 2)
    x2 = t.var(1.0)
    v1 = t.sin(x1)
    v2 = t.add(x1, x2)
    v3 = t.mult(v1, v2)
    assert t.tape[v3].value == 1.0 + math.pi / 2
    t.backprop(v3)
    print("\n", t)
```

## What Is Next

The program above uses only VJP.  However, modern deep learning systems often use both JVP and VJP to achieve optimal performance, when working with tensors rather than scalar variables.
