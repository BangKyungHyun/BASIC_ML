#!pip install matplotlib seaborn pandas sklearn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 실습에서 사용할 데이터셋을 불러옴
from sklearn.datasets import load_boston
boston = load_boston()

# 보스턴 주택 가격 데이터셋을 대한 설명이 자세하게 출력됨
print(boston.DESCR)

# 데이터셋은 506개의 샘플을 가지고 있으며 13개의 속성들과 이에 타깃값(label)을 갖고 있음
# 간단한 탐험적 데이터 분석을 위해 판다스 데이터 프레임으로 변환 후에 데이터 일부를 확인함
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["TARGET"] = boston.target
df.tail()
print("df.tail()\n", df.tail())

# 각 속성의 분포와 속성 사이의 선형적 관계 유무를 파악하기 위해 페어플롯을 그림
sns.pairplot(df)
plt.show()

# Target 속성에 대응하는 맨 마지막을 줄을 살펴보면 일부 속성들이 Target 속성과
# 약간의 선형적 관계를 띄는 것을 볼 수 있음
# 선형적 관계를 띄는 것으로 보이는 일부 속성을 추려 내여 다시 페어플롯을 그림
# 그림의 맨 첫 줄이 target 속성과 대응하여 그린 것임
cols = ["TARGET", "INDUS", "RM", "LSTAT", "NOX", "DIS"]

df[cols].describe()

sns.pairplot(df[cols])
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Numpy 데이터를 파이토치 실수형 텐서로 변환함
data = torch.from_numpy(df[cols].values).float()

print("data.shape = ", data.shape)

# 데이터를 입력 x와 출력 y로 나눔
y = data[:, :1]
x = data[:, 1:]

print("x.shape = ", x.shape, "y.shape = ", y.shape)

# 학습에 필요한 설정값을 정함

n_epochs = 2000
learning_rate = 1e-3
print_interval = 100

# 모델을 생성. 텐서x의 마지막 차원의 크기를 선형 계층의 입력 크기로 주고,
# 텐서 y의 마지막 차원의 크기를 선형 계층의 출력 크기로 함
model = nn.Linear(x.size(-1), y.size(-1))

model

# Instead of implement gradient equation,
# we can use <optim class> to update model parameters, automatically.
# 옵티마이져 생성. 파이토치에서 제공하는 옵티마이저 클래스를 통해 최적화 작업을 수행
# backward함수를 호출한 후 옵티마이저 객체에서 step 함수를 호출하면 경사하강을
# 1회 수행한다.
optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate)

# 정해진 에폭만큼 for 반복문을 통해 최적화를 수행
for i in range(n_epochs):
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i + 1) % print_interval == 0:
        print('Epoch %d: loss=%.4e' % (i + 1, loss))

# 결과 확인
# 모델을 통과한 y_hat를 가져와서 실제 y와 비교하기 위한 페어 플롯을 그림 
df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(),
                  columns=["y", "y_hat"])

# 왼쪽 위에 그려진 y의 분포와 오른쪽 아래에 그려진 y_hat의 분포가 약간은 다르게 나타난 것을
# 볼수 있음. 하지만 오른쪽 위에 그려진 y와 왼쪽 아래의 y_hat과의 비교에서는 대부분의 점들이
# 빨간색 점선 부근에 나타나 있는 것을 확인할 수 있음
sns.pairplot(df, height=5)
plt.show()