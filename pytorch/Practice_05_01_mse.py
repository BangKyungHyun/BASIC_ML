import torch

################################################################################
# STEP 1. Mean Square Error Loss 를 함수를 이용하여 직접 구현
################################################################################

def mse(x_hat, x):

  # |x_hat| = (batch_size, dim)
  # |x| = (batch_size, dim)
  y = ((x-x_hat)**2).mean()

  return y

x = torch.FloatTensor([[1,1],
                       [2,2]])

x_hat = torch.FloatTensor([[0,0],
                           [0,0]])

print("x.size() =", x.size())
# x.size() = torch.Size([2, 2])

print("x_hat.size() =", x_hat.size())
# x_hat.size() = torch.Size([2, 2])

# mse 공식 : (x-x_hat)**2 / 4
mst_value = mse(x_hat, x)
# (1-0)**2 + (1-0)**2 + (2-0)**2 + (2-0)**2 = 1 + 1 + 4 + 4 = 10 /4
print("mst_value = ", mst_value)
# mst_value =  tensor(2.5000)

################################################################################
# STEP 2. torch.nn.functional 사용하기
################################################################################
# PyTorch 내장 MSE 손실 함수 활용
import torch.nn.functional as F

# reduction parameter의 default 값은 'mean',
# 즉 정답과 예측값의 cross entropy를 한 후 나온 값들의 평균을 내서 return
F.mse_loss(x_hat,x)
print("F.mse_loss(x_hat,x) =", F.mse_loss(x_hat,x))
# F.mse_loss(x_hat,x) = tensor(2.5000)

# reduction = 'sum'을 해주게 되면 합의 값을 return
F.mse_loss(x, x_hat, reduction='sum')
print("F.mse_loss(x, x_hat, reduction='sum') =", F.mse_loss(x, x_hat, reduction='sum'))
# F.mse_loss(x, x_hat, reduction='sum') = tensor(10.)

# reduction = 'none'을 해주게 되면 평균을 return 해주는 것이 아닌
# 모든 pixel 의 cross entropy 결과 값을  return

# mean 값이 아닌 모든 pixel의 결과 차이 값을 return 값이 필요할 때도 있을 수
# 있기 때문에 그럴 때는 reduction = 'none'으로 설정해 주면 됨
F.mse_loss(x_hat, x, reduction='none')
print("F.mse_loss(x_hat, x, reduction='none') =", F.mse_loss(x_hat, x, reduction='none'))
# F.mse_loss(x_hat, x, reduction='none') = tensor([[1., 1.],
#         [4., 4.]])

################################################################################
# STEP 3. torch.nn 사용하기
################################################################################

import torch.nn as nn

mse_loss = nn.MSELoss()
mse_loss(x_hat, x)
print(" mse_loss(x_hat, x) = ", mse_loss(x_hat, x))
# mse_loss(x_hat, x) = tensor(2.5000)