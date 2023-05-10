#!pip install matplotlib seaborn pandas sklearn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 데이터셋 호출
from sklearn.datasets import load_boston
boston = load_boston()

# 보스턴 주택 가격 데이터셋에 대한 설명이 자세히 출력
print(boston.DESCR)

# 이 데이터셋은 506개의 샘플을 가지고 있으며 13개의 속성들과 이에 대한 타킷값(레이블)을 가지고 있음
# 간단한 탐험적 데이터 분석을 위해서 판다스 데이터 프레임으로 변환 후에 데이터 일부를 확인한다.
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["TARGET"] = boston.target
df.tail()

# 각 속성의 분포와 속성 사이의 선형적 관계 유무를 판단하기 위해서 페어플롯(pair plot)을 그려봅니다.
# Target 속석에 대응되는 맨 마지막 줄을 살펴보면 일부 속성들이 Targer 속성과 약간의 선형적 관계를
# 띄는 것을 볼 수 있음
sns.pairplot(df)
plt.show()

# 선형적 관계를 띄는 것으로 보이는 일부 속성을 추려 다시 페어플롯을 그려 봄
# 이번에는 그림의 맨 첫 줄이 Target 속성과 대응하여 그런 것임
cols = ["TARGET", "INDUS", "RM", "LSTAT", "NOX", "DIS"]

df[cols].describe()

sns.pairplot(df[cols])
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Numpy 데이터를 파이토치 실수형 텐서로 변환
data = torch.from_numpy(df[cols].values).float()

data.shape
# 데이터를 입력 x와 출력 y로 나눔 -- Split x and y.
y = data[:, :1]
x = data[:, 1:]

print(x.shape, y.shape)

# 학습에 필요한 설정값을 정함
n_epochs = 2000
learning_rate = 1e-3
print_interval = 100

# 모델을 생성. 텐서x의 마지막 차원의 크기를 선형 계층의 입력 크기로 주고, 텐서 y의 마지막 차원의 \
# 크기를 선형 계층의 출력 크기로 함
model = nn.Linear(x.size(-1), y.size(-1))

model

# Instead of implement gradient equation,
# we can use <optim class> to update model parameters, automatically.
# 옵티마이저를 생성. 파이토치에서 제공하는 옵티마이저 클래스를 통해 해당작업을 대신 수행
# backward함수를 호출한 후 옵티마이저 객체에서 step함수를 호출하면 경사하강을 1번 수행

optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate)

# epoch만큼 for 반복문을 통해 최적화를 수행
for i in range(n_epochs):
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i + 1) % print_interval == 0:
        print('Epoch %d: loss=%.4e' % (i + 1, loss))

# 결과 확인
# 모델을 통과한 y_hat을 가져와 실제 y와 비교하기 위한 페어플롯을 그림
df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(),
                  columns=["y", "y_hat"])

sns.pairplot(df, height=5)
plt.show()