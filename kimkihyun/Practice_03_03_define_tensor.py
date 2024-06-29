import torch
# Tensor Allocation

ft = torch.FloatTensor([[1, 2],
                        [3, 4]])
print('torch.FloatTensor = \n', ft);

lt = torch.LongTensor([[1, 2],
                       [3, 4]])
print('torch.LongTensor = \n', lt);

bt = torch.ByteTensor([[1, 0],
                       [0, 1]])
print('torch.ByteTensor = \n', bt);

x = torch.FloatTensor(3, 2)
print('torch.FloatTensor = \n',x);

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