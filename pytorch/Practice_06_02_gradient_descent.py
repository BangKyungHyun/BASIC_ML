import torch
import torch.nn.functional as F

target = torch.FloatTensor([[.1,.2,.3],
                            [.4,.5,.6],
                            [.7,.8,.9]])

x = torch.rand_like(target)
# This means the final scalar will be differentiate by x.
# 이는 최종 스칼라가 x로 미분됨을 의미합니다.

x.requires_grad = True
# You can get gradient of x, after differentiation.
# 미분 후 x의 기울기를 얻을 수 있습니다.

print("x =", x)

loss = F.mse_lose(x, target)
print ("loss =", loss)
