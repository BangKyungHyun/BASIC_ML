import torch

def describe(x):
    print("타입 : {} ".format(x.type()))
    print("크기 : {} ".format(x.shape))
    print("값 : \n{} ".format(x))

# 파이토치에서 torch Tensor로 텐서만들기
print(describe(torch.Tensor(2,3)))

# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[2.9121e+20, 6.1377e-43, 2.9121e+20],
#         [6.1377e-43, 3.2177e+20, 6.1377e-43]])
# None

# 랜덤하게 초기화된 텐서 만들기

print(describe(torch.rand(2,3))) # 균등분포
# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[0.2532, 0.8317, 0.6183],
#         [0.8257, 0.0735, 0.6882]])
# None

# 표준 정규분포 : 평균이 0이고 분산이 1인 정규분포
print(describe(torch.randn(2,3)))
# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[ 0.8619,  0.5900,  0.5366],
#         [-1.0311,  0.4991,  0.0028]])
# None

# fill_() 메서드 사용하기
print(describe(torch.zeros(2,3)))
# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
x = torch.ones(2,3)
print(describe(x))
# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
x.fill_(5)
# print(describe(x))
# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[5., 5., 5.],
#         [5., 5., 5.]])

# 파이썬 리스트로 텐서를 만들고 초기화하기
x = torch.Tensor([[1,2,3],
                  [4,5,6]])
describe(x)

# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])


# 넘파이로 텐서를 만들고 초기화하기
import numpy as np
npy = np.random.rand(2,3)
describe(torch.from_numpy(npy))

# 타입 : torch.DoubleTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[0.2150, 0.8684, 0.7213],
#         [0.9308, 0.6315, 0.1603]], dtype=torch.float64)

# 텐서 타입과 크기
x = torch.FloatTensor([[1,2,3],
                       [4,5,6]])
describe(x)

# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

x = x.long()
describe(x)

# 타입 : torch.LongTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[1, 2, 3],
#         [4, 5, 6]])

x = torch.tensor([[1,2,3],
                  [4,5,6]], dtype=torch.int64)
describe(x)

# tensor([[1, 2, 3],
#         [4, 5, 6]])
# 타입 : torch.LongTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[1, 2, 3],
#         [4, 5, 6]])

x = x.float()
describe(x)

# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

# 텐서 연산
x = torch.randn(2,3)
describe(x)

# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[ 0.5460,  0.6137,  0.4467],
#         [-1.6872,  0.5612,  0.9906]])
# 타입 : torch.FloatTensor

describe(torch.add(x,x))
# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[ 1.0920,  1.2274,  0.8934],
#         [-3.3744,  1.1224,  1.9812]])
describe( x + x)

# 타입 : torch.FloatTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[ 1.0920,  1.2274,  0.8934],
#         [-3.3744,  1.1224,  1.9812]])

# 차원별 텐서 연산

x = torch.arange(6)
describe(x)
# 타입 : torch.LongTensor
# 크기 : torch.Size([6])
# 값 :
# tensor([0, 1, 2, 3, 4, 5])
x = x.view(2,3)
describe(x)
# 타입 : torch.LongTensor
# 크기 : torch.Size([2, 3])
# 값 :
# tensor([[0, 1, 2],
#         [3, 4, 5]])

