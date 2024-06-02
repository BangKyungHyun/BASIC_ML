################################################################################
# STEP 1. torch.nn.Module 클래스 상속 받기
################################################################################

# nn.Module을 상속받은 MyLinear라는 클래스 정의
# nn.Module을 상속받은 클래스는 보통 2개의 메서드 __init__과 forward를 오버라이드
# __init__ 함수는 계층 내부에서 필요한 변수를 미리 선언하는 부분이며,
# 또 다른 계층(nn.Module을 상속받은 클래스의 객체)을 소유하도록 할 수도 있음
# forward 함수는 계층을 통과하는데 필요한 계산을 수행하도록 구현하는 부분
import torch
import torch.nn as nn

x = torch.FloatTensor([[1, 1, 1],
                       [2, 2, 2],
                       [3, 3, 3],
                       [4, 4, 4]])

W = torch.FloatTensor([[1, 2],
                       [3, 4],
                       [5, 6]])

b = torch.FloatTensor([2,2])

print('\n00000000000000000000000\n')

class Mylinear(nn.Module):

    print('def __init__ 1111 ')
    def __init__(self, input_dim=3, output_dim=2):

        print('def __init__ begin')

        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.W = torch.FloatTensor(input_dim, output_dim)
        self.b = torch.FloatTensor(output_dim)

        print('__init__ self.W = ', self.W)
        print('__init__ self.b = ', self.b)

        print('def __init__ end')

    print('forward 11111')

    def forward(self, x):

        print('def forward begin')

        # |x| = (batch_size, input_dim)
        y = torch.matmul(x, self.W) + self.b
        # |y| = (batch_size, input_dim) * (input_dim, output_dim)
        #     = (batch_size, output_dim)

        print('forward self.W = ', self.W)
        print('forward self.b = ', self.b)
        # print('x      = ', x)
        print('y      = ', y)

        print('forward end')
        return y

print('\n11111111111111111111111\n')

print('Mylinear(3,2) = ', Mylinear(3,2))

print('\n22222222222222222222222\n')

linear = Mylinear(3,2)

print('linear        = ', linear)

print('\n333333333333333333333333\n')

print('linear(x) = ', linear(x))

# 여기서 중요한 점은 forward 함수를 따로 호출하지 않고 객체명에 바로 괄호를 열러서 텐서 x를
# 인수로 넘겨 주었다는 점이다. 이처럼 nn.Module의 상속닫은 객체는 __call__ 함수와 forward가
# 매핑되어 있어서 forward를 직접 부를 필요가 없음

y = linear(x)

print('y = ', y)

print('W = ', W)
print('b = ', b)

print('\n444444444444444444444444\n')

# 여기까지 우리는 nn.Module을 상속받아 선형 계층을 구성하여 보았습니다.
# 하지만 이 방법도 아직은 제대로 된 방법이 아닙니다.
# 왜냐하면 파이토치 입장에서는 비록 MyLinear라는 클래스의 계층으로 인식하고 계산도 수행하지만,
# 내부에 학습할 수 있는 파라미터는 없는것으로 인식하기 때문입니다.
# 예를 들어 다음의 코드를 실행하면 아무것도 출력되지 않을 것입니다.

print('linear.parameters() = ', linear.parameters())
for p in linear.parameters():
    print('\n55555555555555555555555555\n')

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