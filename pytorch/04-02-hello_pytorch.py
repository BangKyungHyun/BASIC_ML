import torch

a = torch.FloatTensor([[1, 2],
                       [3, 4]])
b = torch.FloatTensor([[1, 2],
                       [1, 2]])

c = torch.matmul(a, b)
print(c);