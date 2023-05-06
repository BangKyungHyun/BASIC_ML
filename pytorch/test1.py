import torch
import numpy as np

ft = torch.FloatTensor([[1,2],[3,4]])

lt = torch.LongTensor([[1,2],[3,4]])

bt = torch.ByteTensor([[1,0],[0,1]])

x = torch.FloatTensor(3,2)

y = np.array([[1,2],[3,4]])

print(ft)

print(lt)

print(bt)

print(x)

print(y, type(y))