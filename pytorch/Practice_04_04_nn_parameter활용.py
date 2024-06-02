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

        # 제대로 된 방법은 W와 b를 파이토치에서 학습 가능하도록 인식할 수 있는 파라미터로
        # 만들어 주어야 합니다.
        # 이것은 torch.nn.Parameter 클래스를 활용하면 쉽게 가능합니다.
        # 다음의 코드와 같이 파이토치 텐서 선언 이후에 nn.Parameter로 감싸주면 됩니다.

        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.b = nn.Parameter(torch.FloatTensor(output_dim))

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

y = linear(x)

print('y = ', y)

print('W = ', W)
print('b = ', b)

print('\n444444444444444444444444\n')

# 그럼 다음과 같이 파라미터를 출력하도록 했을 때, 파라미터가 정상적으로 출력되는 것을 볼 수 있습니다.

print('linear.parameters() = ', linear.parameters())
for p in linear.parameters():
    print('\n55555555555555555555555555\n')

    print('p =', p)