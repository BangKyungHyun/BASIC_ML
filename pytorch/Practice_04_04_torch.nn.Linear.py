import torch

# 3×2 크기의 행렬 W와 2개의 요소를 갖는 벡터 b 선언
# 이 텐서들을 파라미터로 삼아, 선형 계층 함수 구성

W = torch.FloatTensor([[1, 2],
                       [3, 4],
                       [5, 6]])
b = torch.FloatTensor([20,20])

def linear(x, W, b):
    z = torch.matmul(x, W) + b

    return z

# 3개의 요소를 갖는 4개의 샘플을 행렬로 나타내면 x와 같이 4×3  크기의 행렬이 될 것입니다.
# 이것을 다음 코드와 같이 함수를 활용하여 선형 계층을 통과시킬 수 있습니다.
x = torch.FloatTensor(4,3)

yy = linear(x, W ,b)

print("W = ",W)
print("b = ",b)
print("x = ",x)
# print("y = ",y)
print("yy = ",yy)

print("yy.size() = ",yy.size())


# 파이토치에는 nn(neural networks) 패키지가 있고, 내부에는 미리 정의된 많은 신경망들이 들어있습니다.
# 그리고 그 신경망들은 torch.nn.Module 이라는 추상클래스를 상속받아 정의되어 있습니다.
# 이 추상클래스를 상속받아 선형 계층을 구현할 수 있습니다.
# 그에 앞서 torch.nn 패키지를 불러옵니다

import torch.nn as nn

#linear = nn.linear(3,2)

# nn.Module을 상속받은 MyLinear라는 클래스 정의
# nn.Module을 상속받은 클래스는 보통 2개의 메서드 __init__과 forward를 오버라이드
# __init__ 함수는 계층 내부에서 필요한 변수를 미리 선언하는 부분이며,
# 심지어는 또 다른 계층(nn.Module을 상속받은 클래스의 객체)을 소유하도록 할 수도 있음
# forward 함수는 계층을 통과하는데 필요한 계산을 수행하도록 구현하는 부분

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