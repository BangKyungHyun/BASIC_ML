import torch

# Mean Square Error (MSE) Loss
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
print("mst_value = ", mst_value)
# mst_value =  tensor(2.5000)

# Predefined MSE in PyTorch
import torch.nn.functional as F

F.mse_loss(x_hat,x)
print("F.mse_loss(x_hat,x) =", F.mse_loss(x_hat,x))

F.mse_loss(x, x_hat, reduction='sum')
print("F.mse_loss(x, x_hat, reduction='sum') =", F.mse_loss(x, x_hat, reduction='sum'))

F.mse_loss(x_hat, x, reduction='none')
print("F.mse_loss(x_hat, x, reduction='none') =", F.mse_loss(x_hat, x, reduction='none'))

import torch.nn as nn

mse_loss = nn.MSELoss()

mse_loss(x_hat, x)
print(" mse_loss(x_hat, x) = ", mse_loss(x_hat, x))


