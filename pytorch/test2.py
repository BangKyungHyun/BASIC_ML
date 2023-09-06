import torch
import torch.nn as nn

m = nn.Linear(2, 300)
print('m =', m)
input = torch.randn(4, 2)
print('input =', input)
print('m(input) =', m(input))


output = m(input)
print('output =', output)

print('output.size() = ', output.size())
