################################################################################
# STEP 1. torch.nn.Module 클래스 상속 받기
################################################################################

# 파이토치에는 nn(neural networks) 패키지가 있고, 내부에는 미리 정의된 많은 신경망들이 들어 있음
# 그리고 그 신경망들은 torch.nn.Module 이라는 추상클래스를 상속받아 정의되어 있음
# 이 추상클래스를 상속받아 선형 계층을 구현할 수 있음

import torch.nn as nn

class Mylinear(nn.Module):

    def __init__(self, input_dim=3, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.W = torch.FloatTensor(input_dim, output_dim)
        self.b =  torch.FloatTensor(output_dim)

    def forward(self,x):
        # |x| = (batch_size, input_dim)
        y = torch.linear(x)
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