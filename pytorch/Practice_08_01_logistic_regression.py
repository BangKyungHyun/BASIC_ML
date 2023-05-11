import torch
import torch.nn as nn

from matplotlib import pyplot as plt

x = torch.sort(torch.randn(100) * 10)[0]

print('x =', x )