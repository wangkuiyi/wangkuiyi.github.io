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
