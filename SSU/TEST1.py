import numpy as np
import torch
from torch import nn
import random
# m = nn.Linear(2, 3)
# # input = torch.randn(4, 2)
# input = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
#
# output = m(input)
#
# print('m.weight =', m.weight)
# print('m.weight.size() =', m.weight.size())
#
#
# print('m.bias =', m.bias)
# print('m.bias.size() =', m.bias.size())
#
# print('input =', input)
# print('output =', output)
#
# print('input.size() =',input.size())
# print('output.size() =',output.size())
#
input_length = 10
teacher_forcing_ratio = 0.5

for i in range(input_length):
    # teacher_force = random.random() < teacher_forcing_ratio
    teacher_force = random.random()
    print('teacher_force  = ', teacher_force)
    print('random.random() = ', random.random())
    print('teacher_forcing_ratio = ', teacher_forcing_ratio)
