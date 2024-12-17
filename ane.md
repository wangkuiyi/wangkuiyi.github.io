# Use Conv2D and Matmal on Apple Neural Engine

In 2022, some Apple engineers published a blog post [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers).  Apple Nerual Engine (ANE) is part of the Apple silicon chip that exists in every iPhone, iPad, and Mac computer.  In less than two years since the publication, there comes Apple Intelligence.  As explained in a recent blog post [Introducing Apple’s On-Device and Server Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models), the on-device model plays an important role in Apple Intelligence.

In the 2022 blog post, the authors explained several principles to accelerate Tranformer models using ANE.  The number one is titled *Principle 1: Picking the Right Data Format*.  In this section, the authors explains

> We represent this tensor in the (B, C, 1, S) data format because the most conducive data format for the ANE (hardware and software stack) is 4D and channels-first.

> To migrate to the desirable (B, C, 1, S) data format, we swap all nn.Linear layers with nn.Conv2d layers.

Reading the [source code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py) accompanying with the blog post, we see that all `nn.Linear` layers are replaced by `nn.Conv2d`.

This article explains why we can use `nn.Conv2d` to replace `nn.Linear`.  Before diving into the details, let us quickly verify that the two layers are exchangable.

```python
import torch
import torch.nn as nn

B = 2
I = 5  # input features or channels
O = 3  # output features or channels
K = 1

# Create two layers that have the same parameter values.
linear = nn.Linear(in_features=I, out_features=O)
conv = nn.Conv2d(in_channels=I, out_channels=O, kernel_size=1)
with torch.no_grad():
    conv.weight.data = linear.weight.data.view(O, I, 1, 1)
    conv.bias.data = linear.bias.data

# Verify they product identical outputs given the same input.
x = torch.randn(B, I)
linear_output = linear(x)
conv_output = conv(x.view(B, I, 1, 1)).view(B, O)
assert torch.allclose(linear_output, conv_output)
```
