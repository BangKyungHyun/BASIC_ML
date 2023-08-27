import torch

################################################################################
# STEP 1.  행렬 곱 연산                                                          #
################################################################################
print('\nSTEP 1.  행렬 곱 연산\n')

a = torch.FloatTensor([[1, 2],
                       [3, 4]])
b = torch.FloatTensor([[1, 2],
                       [1, 2]])

c = torch.matmul(a, b)
print(c);

# tensor([[ 3.,  6.],
#         [ 7., 14.]])

################################################################################
# STEP 2.  선형 계층 직접 구현하기                                                #
################################################################################

print('\nSTEP 2.  선형 계층 직접 구현하기\n')

# 3×2 크기의 행렬 W와 2개의 요소를 갖는 벡터 b 선언
# 이 텐서들을 파라미터로 삼아, 선형 계층 함수 구성

W = torch.FloatTensor([[1, 2],
                       [3, 4],
                       [5, 6]])
b = torch.FloatTensor([2,2])


print('W.size = ', W.size())
print('W      = ', W)
print('b.size = ', b.size())
print('b      = ', b)

def linear(x, W, b):
    z = torch.matmul(x,W) + b
    print('z      = ', z)

    return z

# 3개의 요소를 갖는 4개의 샘플을 행렬로 나타내면 x와 같이 4×3 크기의 행렬이 됨
# 이것을 다음 코드와 같이 함수를 활용하여 선형 계층을 통과시킬 수 있음

x = torch.FloatTensor([[1, 1, 1],
                       [2, 2, 2],
                       [3, 3, 3],
                       [4, 4, 4]])

print('x.size = ', x.size())
print('x      = ', x)

print('111111111')
yy = linear(x, W ,b)
print('222222222')
yy = linear(x, W ,b)
print('333333333')
yy = linear(x, W ,b)
print('444444444')

print("yy = ",yy)
print("yy.size() = ",yy.size())