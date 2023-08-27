

print("---- import torch.nn as nn ----- ")

# nn.Module을 상속받은 MyLinear라는 클래스 정의
# nn.Module을 상속받은 클래스는 보통 2개의 메서드 __init__과 forward를 오버라이드
# __init__ 함수는 계층 내부에서 필요한 변수를 미리 선언하는 부분이며,
# 심지어는 또 다른 계층(nn.Module을 상속받은 클래스의 객체)을 소유하도록 할 수도 있음
# forward 함수는 계층을 통과하는데 필요한 계산을 수행하도록 구현하는 부분

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

        print('self.W = ', self.W)
        print('self.b = ', self.b)
        # print('x      = ', x)
        print('y      = ', y)

        return y

linear = Mylinear(3,2)
print('Mylinear(3,2) = ', Mylinear(3,2))
print('linear        = ', linear)

y = linear(x)
print('x = ', x)

print('W = ', W)
print('b = ', b)
print('y = ', y)

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