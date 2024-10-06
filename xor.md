# Train An MLP Using Your Brain Rather Than GPU to Address the XOR Classification Challenge

This is a great mental exercise to understand why do we need activation functions.

Being asked this quesiton in a job interview, you are supposed to give an example classification challenge, which

1. is not linearly classifiable,
1. can be solved by an multilayer perceptron,
1. with an activation function.

A well-known example is the XOR challenge. Surprisingly, when I Googled about it, some results [1](https://dev.to/jbahire/demystifying-the-xor-problem-1blk) [2](https://priyansh-kedia.medium.com/solving-the-xor-problem-using-mlp-83e35a22c96f) [3](https://dataqoil.com/2022/06/24/multilayer-percepron-using-xor-function-from/) claim that backpropagation algorithm could estimate the parameters of MLP that can solve the challenge, but didn't give the estimated parameters. Some [4](https://stackoverflow.com/questions/37734655/neural-network-solving-xor) suggests that we need to build an MLP to mimic the logical equation `x1 XOR x2 == NOT (x1 AND x2) AND (x1 OR x2)`, but didn't explain how. A few of them present wrong example parameters. This inspired my attempt to address this.

## What is Linearly Classifiable

Consider some points on the 2D plane. Each point is in either of two colors, say, blue and red.  The definitons is something like -- if we could draw a straightline to make sure that red points are on one side of the line and blue points are on the other side, we say these colored points are linearly classifiable.

For programmers, a definition makes sense only if it is computable.  The above definition makes sense because the linear regression model, when estimated using the error backpropagation algorithm given the coordinate and color of each and every point, tells where the line lies.

Consider that all red points have their y coordinate larger than 10, and blue points all have y less than 10, then the line $y=10$ is a perfect line to separate them by color.

## XOR Challenge is Not Linearly Classifiable

In the XOR challenge, there are four points, whose coordinates and colors are as follows:

- (0,0) - red
- (0,1) - blue
- (1,0) - blue
- (1,1) - red

It is not too hard to excersise your brain to imagine a line on the 2D plane that goes across any place. However, you rotate the line, there is no way to separate the four points by their colors.

## Transforming XOR into a Easier Form

With the above imagination exercise, it also not hard to realize that if we could transform the point (0,1) to where at or close to (1,0), we could draw a line to classify. Or, alternatively, if we transform (1,1) to be close to (0,0).

A very simple form of transformation is linear transformation, which is defind by a matrix $W$ and the corresponding bias vector $v$ 

$$W=[[w_{11}, w_{12}], [w_{21}, w_{22}]]$$
$$v=[v_1, v_2]^T$$

Transforming the point $(x,y)$ moves it to the new coordinate:

$$ (x,y) \rightarrow ( w_{11}x + w_{12} y + v_1, w_{12} x + w_{22} y + v_2) $$

As we want to guess these parameters without running the backpropagation algorithm, it is always good to start with simple guesses.

Let us begin with that both $W$ and $b$ are zeros. This would fuse all points to $(0,0)$ and thus make them completely inclassifiable.

If $W$ is diagonally identical and $b$ is zero, the transformation wouldn't move any point, thus it does not ease the challenge.

Then let us guess that $W$ contains all 1's.  In this case, the transformation results would be:

$$ (x,y) \rightarrow ( x + y + v_1, x + y + v_2) $$

Or, for the XOR challenge:

$$ (0,0) \rightarrow ( v_1, v_2) $$
$$ (0,1) \rightarrow ( 1 + v_1, 1 + v_2) $$
$$ (1,0) \rightarrow ( 1 + v_1, 1 + v_2) $$
$$ (1,1) \rightarrow ( 2 + v_1, 2 + v_2) $$

The middle two points (both blue colored) are now at the same coordinate. This is a good sign! 

However, this is not good enough. Because $(v_1,v_2)$ is like moving from $(0,0)$ for an offset $(v_1, v_2)$, and similarly, the rest points are moving from $(1,1)$ and $(2,2)$ for the same offset. This would make all new points on the same line and the blue ones are in between the two red ones. This setup is not linearly classifiable.

Here is where we need activation functions, or, the non-linearity, to move these points away from the line on which they are.

## We Need Activation Functions

Some well-known activation functions include softmax, tanh, and ReLU. All of them clamp negative inputs to 0 or a close-to-0 value. The keyword here is "clamp"!

Given the three coordinates after the above linear transformation:

$$ ( v_1, v_2) $$
$$ ( 1 + v_1, 1 + v_2) $$
$$ ( 2 + v_1, 2 + v_2) $$

If by carefully choosing the values of $v_1$ and $v_2$, we can make the activation function clamp both coordinates of a point, one coordinates of another, and no coordinates of the third, then we should be able to move the three points away from the line that they share.

The simplest guess is $v_1=v_2$. Unfortunately, that would keep the three points on the line $y=x$ or some line that in parallel, denoted by $y=x+\xi$.

As long as both $v_1$ and $v_2$ are negative, the first point would be clamped back to $(0,0)$. It is then the "a point" as mentioned above. If the absolute values of $v_1$ and $v_2$ are less than 2, the point $(2+v_1,2+v_2)$ wouldn't be clamped, so it would be the third point as mentioned above. Now, given the two coordinate values of the point $(1+v_1,1+v_2)$, we want one of them negative and the other one positive. To do so, we could have

$$ v_1 = -0.5 $$
$$ v_2 = -1.5 $$

## Now, Linear Classify!

The above linear transformation and activation moves the original four points as follows:

$$ (0,0) \rightarrow \sigma( v_1, v_2) \rightarrow (0, 0) $$
$$ (0,1) \rightarrow \sigma( 1 + v_1, 1 + v_2) \rightarrow (0.5, 0) $$
$$ (1,0) \rightarrow \sigma( 1 + v_1, 1 + v_2) \rightarrow (0.5, 0) $$
$$ (1,1) \rightarrow \sigma( 2 + v_1, 2 + v_2) \rightarrow (1.5, 0.5) $$

It is not too hard to tell that the following line could separate the red points from the blue ones:

$$ y = \frac{0.5}{1.5} x - 0.001 = \frac{1}{3}x - 0.001 $$

where the slope is derived from the third coordinate because the first two are on x-axis now. The intercept $-0.001$ is an arbitrary small enough negative value.

## The MLP for the XOR Classification

A MLP is defined as folows. Given a point at coordinate $(x,y)$, the MLP runs the following equation:

$$ s = \sigma( v_1 h_1 + v_2 h_2 + c) $$
$$ h_1 = \sigma( w_{11} x + w_{12} y + b_1 ) $$
$$ h_2 = \sigma( w_{21} x + w_{22} y + b_2) $$

The above derivation gave us an estimation of the parameters that can solve the XOR classification challenge:

$$ s = \sigma( 3 h_1 +  h_2 - 0.001 ) $$
$$ h_1 = \sigma( x + y - 0.5 ) $$
$$ h_2 = \sigma( x + y - 1.5 ) $$

If $s>0$, the point $(x,y)$ is red; otherwise, it is blue.
