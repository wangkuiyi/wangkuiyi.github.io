# Using Conv2D for Linear Projection on Apple Neural Engine

If you are interested in Apple Intelligence, you may wanted to check out the 2022 blog post [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers) and the more recent [Deploying Attention-Based Vision Transformers to Apple Neural Engine](https://machinelearning.apple.com/research/vision-transformers).  Both blog posts feature open sourced code that demonstrates the use of Conv2d layers as replacements for linear projection:

1. [DistilBERT implementation](https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/huggingface/distilbert.py)
2. [Vision Transformers implementation](https://github.com/apple/ml-vision-transformers-ane/blob/main/vision_transformers/attention_utils.py)

The 2022 blog post introduces this replacement as *Principle 1: Picking the Right Data Format* of improving the Apple Neural Engine (ANE) performance.

## Empirical Verification

The code snippet below demonstrates the interchangeability between `nn.Conv2d` and `nn.Linear`.  When projecting a batch of $B$ $I$-dimensional vectors into the $O$-dimensional space, you can either:

1. use a traditional $O\times I$ linear projection matrix, or
2. reshape the input batch to shape $(B, I, 1, 1)$ and use a Conv2d layer with a kernel of shape $(O, I, 1, 1)$.

The output of the Conv2d operation will have the shape $(B, O, 1, 1)$, which can be reshaped to $(B, O)$, yielding results identical to the linear projection. This equivalence holds whether or not a bias term is included.

```python
import torch
import torch.nn as nn

# Define batch size, input, and output dimensions
B = 2
I = 5
O = 3

# # Create two layers with identical parameters.
linear = nn.Linear(in_features=I, out_features=O)
conv = nn.Conv2d(in_channels=I, out_channels=O, kernel_size=1)
with torch.no_grad():
    conv.weight.data = linear.weight.data.view(O, I, 1, 1)
    conv.bias.data = linear.bias.data

# Verify that both layers produce identical outputs.
x = torch.randn(B, I)
linear_output = linear(x)
conv_output = conv(x.view(B, I, 1, 1)).view(B, O)
assert torch.allclose(linear_output, conv_output)
```

## How Conv2d Works

The simplest form of Conv2d outputs a grayscale image where each pixel is the weighted average of an $m\times n$ region in the input $m\times n$ grayscale image.  The $m\times n$ weights are referred to as the kernel of the Conv2d operation.  If $m=n=1$, the kernel contains only a single scalar weight, and the Conv2d operation effectively scales each pixel of the input image by this weight.

In the generalized form of Conv2d, input images can have multiple channels.  For instance, an image may have three channels for red, green, and blue.  Convolution over a multi-channel image requires a separate kernel for each channel.  Specifically:

1. Each input channel is convolved with its corresponding kernel, resulting in one output image per channel.
2. The $I$-channel outputs are then summed to produce a single output channel.

If the output is also multi-channel (e.g., $O$ channels), the Conv2d operation requires $O$ groups of $I$-channel kernels.  Each kernel group processes the $I$-channel input independently, and the results are aggregated into an $O$-channel output.

## Conv2d As Linear Projection

To linearly project an $I$-dimensional input vector into an $O$-dimensional space using Conv2d, we can reinterpret the vector as an $I$-channel $1\times 1$ image:

1. The $O\times I$ projection matrix is represented as $O$ groups of $I$-channel $1\times 1$ kernels.
2. The Conv2d operation applies these kernels to the input image, producing an $O$-channel $1\times 1$ output image.

When generalized to linear projection of a batch of $B$ input vectors, we interprete the input as $B$ $1\times 1$ images, each with $I$ channels.  The output of Conv2d would be a batch of $B$ $1\times 1$ images, each with $O$ channels.  Then, the equivalence aligns with the explanation in the previous section.

This approach bridges the gap between Conv2d and Linear, allowing the efficient use of convolutional operations for tasks traditionally handled by linear layers on Apple Neural Engine.
