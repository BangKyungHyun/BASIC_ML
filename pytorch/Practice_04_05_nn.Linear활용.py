################################################################################
# STEP 1. nn.Linear 활용하기
################################################################################

# 사실은 위의 복잡한 방법들은 모두 필요 없고, torch.nn에 미리 정의된 선형 계층을 불러다 쓰면
# 매우 간단합니다.

# 다음의 코드는 nn.Linear 를 통해 선형 계층을 활용하는 모습입니다.

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

# 또한 앞서 말한 대로 nn.Module을 상속받아 정의한 나만의 계층 클래스는
# 내부에 nn.Module의 하위 클래스를 소유할 수 있습니다.

class Mylinear(nn.Module):

    print('def __init__ 1111 ')

    def __init__(self, input_dim=3, output_dim=2):
        # 코드는 nn.Module을 상속받아 MyLinear 클래스를 정의하고 있는데,
        # __init__ 함수 내부에는 nn.Linear를 선언하여 self.linear에 저장하는 모습입니다

        print('init begin')

        self.input_dim = input_dim
        self.output_dim = output_dim
        print('init begin 2')

        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        print('def __init__ end')

    print('forward 11111')

    def forward(self,x):

        print('forward begin')

        # forward 함수에서는 self.linear에 텐서 x를 통과시킵니다.
        # 즉, 이 코드도 선형 계층을 구현한 것이라 볼 수 있습니다.[3]
        # [3]: 다만 바깥의 MyLinear는 빈 껍데기에 불과합니다.

        y = torch.linear(x)
        # |x| = (batch_size, input_dim)
        # |y| = (batch_size, input_dim) * (input_dim, output_dim)
        #     = (batch_size, output_dim)

        print('forward self.W = ', self.W)
        print('forward self.b = ', self.b)
        print('y      = ', y)

        print('forward end')

        return y

print('\n11111111111111111111111\n')

linear = nn.Linear(3,2)

print('linear        = ', linear)

print('\n333333333333333333333333\n')

print('linear(x) = ', linear(x))

y = linear(x)

print('y = ', y)
print('x = ', x)
print('W = ', W)
print('b = ', b)

print('\n444444444444444444444444\n')

print('linear.parameters() = ', linear.parameters())
for p in linear.parameters():
    print(p)

    print('\n55555555555555555555555555\n')

    print('p =', p)