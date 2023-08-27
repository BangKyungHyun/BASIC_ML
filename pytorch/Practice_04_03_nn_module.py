import torch

# Tensor Allocation

W = torch.FloatTensor([[1, 2],
                       [3, 4],
                       [5, 6]])
b = torch.FloatTensor([2,2])


print('W.size = ', W.size())
print('W      = ', W)
print('b.size = ', b.size())
print('b      = ', b)

def linear(x, W, b):
    y = torch.matmul(x,W) + b

    return y

x = torch.FloatTensor([[1, 1, 1],
                       [2, 2, 2],
                       [3, 3, 3],
                       [4, 4, 4]])

print('x.size = ', x.size())
print('x      = ', x)

yy = linear(x, W ,b)
print("yy = ",yy)
print("yy.size() = ",yy.size())

import torch.nn as nn

class Mylinear(nn.Module):

    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.W = torch.FloatTensor(input_dim, output_dim)
        self.b = torch.FloatTensor(output_dim)

    def forward(self,x):
        # |x| = (batch_size, input_dim)
        y = torch.matmul(x, self.W) + self.b
        # |y| = (batch_size, input_dim) * (input_dim, output_dim)
        #     = (batch_size, output_dim)

        return y

linear = Mylinear(3,2)

y = linear(x)

for p in linear.parameters():
    print(p)





'''
print(ft);

lt = torch.LongTensor([[1, 2],
                       [3, 4]])
print(lt);

bt = torch.ByteTensor([[1, 0],
                       [0, 1]])
print(bt);

x = torch.FloatTensor(3, 2)
print(x);

# NumPy Compatibility

import numpy as np

# Define numpy array.
x = np.array([[1, 2],
              [3, 4]])
print(x, type(x));

x = torch.from_numpy(x)
print(x, type(x));

x = x.numpy()
print(x, type(x));

# Tensor Type-casting

print(ft.long());

print(lt.float());

print(torch.FloatTensor([1, 0]).byte());

# Get Shape

x = torch.FloatTensor([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]],
                       [[9, 10],
                        [11, 12]]])
# Get tensor shape.
print(x.size());
print(x.shape);

# Get number of dimensions in the tensor.

print(x.dim());
print(len(x.size()));

# Get number of elements in certain dimension of the tensor.

print(x.size(1))
print(x.shape[1])

# Get number of elements in the last dimension.

print(x.size(-1))
print(x.shape[-1])
'''