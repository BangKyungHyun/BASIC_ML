import numpy as np
import torch
from torch import nn

m = nn.Linear(2, 3)
# input = torch.randn(4, 2)
input = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

output = m(input)

print('m.weight =', m.weight)
print('m.weight.size() =', m.weight.size())


print('m.bias =', m.bias)
print('m.bias.size() =', m.bias.size())

print('input =', input)
print('output =', output)

print('input.size() =',input.size())
print('output.size() =',output.size())

