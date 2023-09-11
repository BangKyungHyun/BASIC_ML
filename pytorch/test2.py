import torch
import torch.nn as nn

m = nn.Linear(5, 1)

print('m =', m)
# m = Linear(in_features=5, out_features=1, bias=True)

input = torch.randn(4, 5)
print('input =', input)

# input = tensor([[ 1.1559, -0.9744, -0.4955, -0.0991, -0.7903],
#         [ 0.6457,  0.0207,  0.2133,  0.8778, -0.0765],
#         [ 0.0709,  1.4191, -0.2880,  0.0706,  1.7034],
#         [-1.0156,  0.3434,  0.4634,  0.9138,  1.0603]])

print('m(input) =', m(input))
# m(input) = tensor([[-0.6443],
#         [-0.1951],
#         [ 0.8806],
#         [ 0.8775]], grad_fn=<AddmmBackward0>)

output = m(input)
print('output.size() = ', output.size())

# output.size() =  torch.Size([4, 1])
