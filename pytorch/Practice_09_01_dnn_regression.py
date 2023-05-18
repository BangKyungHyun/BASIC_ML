
# Regression using Deep Neural Networks
# Load Dataset from sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_boston
boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
# 출력값은 TARGET속성으로 저장되도록 함
df["TARGET"] = boston.target
print('df.tail() =\n',df.tail())

# 보스톤 주택 가격 데이터넷은 13개의 속성을 가지며 506개의 샘플로 구성되어 있음.
# 일부 속성만을 활용하여 선형 회귀를 학습했던 것과 달리 이번에는 전체 속성들을 활용하여 
# 심층신경망 학습을 진행함
# 조금 더 쉽고 수월한 최적화 및 성능향샹을 위해 표준 스케줄링을 통해 입력값을 정규화 함
# 보스턴 주택 가격 데이터셋의 각 열이 정규분포를 따른다고 가정하고 표준 스케줄링을 적용함
# 다음 테이블은 표준 스케일링을 적용한 결과를 보여 줌
scalar = StandardScaler()
scalar.fit(df.values[:, :-1])
df.values[:, :-1] = scalar.transform(df.values[:, :-1]).round(4)
print('df.tail() =\n',df.tail())

# Train Model with PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 학습에 필요한 패키지를 불려오고 판다스에 저장된 넘파이 값을 파이토치 텐서로 변환하여
# 입력 텐서 x와 출력 텐서 y를 만듬
data = torch.from_numpy(df.values).float()
print('data.shape = ',data.shape)
# data.shape =  torch.Size([506, 14])

x = data[:, -1:]
y = data[:, :-1]
print('x.shape, y.shape = ',x.shape, y.shape)

n_epochs = 100000
learning_rate = 1e-4
print_interval = 5000

# Build Models
# Build Model using nn.Module

relu = nn.ReLU()
 leaky_relu = nn.LeakyReLU(0.1)

class MyModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.linear1 = nn.Linear(input_dim,3)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(3, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # |x| = (batch_size, input_dim)
        h = self.act(self.linear1(x)) # |h| = (batch_size, 3)
        h = self.act(self.linear2(h))
        h = self.act(self.linear3(h))
        y = self.linear4(h)

        return y

model = MyModel(x.size(-1), y.size(-1))
print('model 1 =  ', model)

# Build Model with LeakyReLU using nn.Sequential

model = nn.Sequential(
    nn.Linear(x.size(-1), 3),
    nn.LeakyReLU(),
    nn.Linear(3, 3),
    nn.LeakyReLU(),
    nn.Linear(3, 3),
    nn.LeakyReLU(),
    nn.Linear(3, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y.size(-1)),
)

print('model  2 = ', model)

optimizer = optim.SGD(model.parameters(),
                      lr = learning_rate)

for i in range(n_epochs):
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if ( i + 1) % print_interval == 0:

        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime,'Epoch %d: loss=%.4e' % (i + 1, loss))

# Let's see the result!