import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# Load Dataset from sklearn
# 위스콘신 유방암 데이터셋은 30개의 속성을 가지며 이를 통해 유방암 여부 예측
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.DESCR)

# 판다스로 데이터를 변환
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target

# Pair plot with mean features
# 각 10개 속성이 평균, 표준편차, 최악값을 나타내고 있기 때문에 속성이 30개임
# 평균과 표준편차, 최악 속성들만 따로 모아서 class속성과 비교하는 페어플롯 표출
sns.pairplot(df[['class'] + list(df.columns[:10])])
plt.show()

# 각 속성별 샘플의 점들이 0번 클래스에 해당하는 경우 아래쪽에 1번 클래스에
# 해당하는 경우 위쪽으로 찍혀 있음
# 만약 이 점들의 클래스별 그룹이 특정값을 기준으로 명확하게 나눠진다면 좋다는 것을 확인
# Pair plot with std features
sns.pairplot(df[['class'] + list(df.columns[10:20])])
plt.show()

# Pair plot with worst features
sns.pairplot(df[['class'] + list(df.columns[20:30])])

# select features
cols = ["mean radius", "mean texture",
        "mean smoothness", "mean compactness", "mean concave points",
        "worst radius", "worst texture",
        "worst smoothness", "worst compactness", "worst concave points",
        "class"]

# 0번 클래스는 파란색, 1번 클래스는 주황색으로 표시. 겹치는 영역이 적을수록 좋은 속성임
for c in cols[:-1]:
    sns.histplot(df, x=c, hue=cols[-1], bins=50, stat='probability')
    #plt.show()

# Train Model with PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df[cols].values).float()
print('data.shape = ', data.shape)
# data.shape =  torch.Size([569, 11])

# 선형회귀와 같이 텐서 x와 텐서 y을 가져옴 
x = data[:, :-1]
y = data[:, -1:]
print('x.shape, y.shape =',x.shape, y.shape)
# x.shape, y.shape = torch.Size([569, 10]) torch.Size([569, 1])

n_epochs = 20000000
learning_rate = 1e-2
print_interval = 200000

# nn.Module을 상속받은 자식 클래스를 정의할 때에는 보통 두개의 함수(메서드)를 오버라이드함. 
# 또한 __init__ 함수를 통해 모델을 구성하는데 필요한 내부모듈(선형계층)을 미리 선언함.
# forward함수는 미리 선언된 내부 모듈을 활용하여 계산을 수행함
class MyModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # |x| = (batch_size, input_dim)
        y = self.act(self.linear(x))
        # |y| = (batch_size, output_dim)

        return y

# 로지스틱 회귀 모델 클래스를 생성하고 BCE 손실함수와 옵티마이저도 준비합니다. 선형회귀와
# 마찬가지로 모델의 입력 크기는 텐서 X의 마지막 차원 크기가 되고 출력크기는 텐서Y의 
# 마지막 크기가 됨
model = MyModel(input_dim=x.size(-1),
                output_dim=y.size(-1))
crit = nn.BCELoss()

optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate)

# 선형회귀와 똑같은 코드로 학습을 진행함
for i in range(n_epochs):
    y_hat = model(x)
    loss = crit(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i + 1) % print_interval == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime,'Epoch %d: loss=%.4e' % (i + 1, loss))

'''
Epoch 10000: loss=2.7718e-01
Epoch 20000: loss=2.2865e-01
Epoch 30000: loss=1.9965e-01
Epoch 40000: loss=1.8072e-01
Epoch 50000: loss=1.6749e-01
Epoch 60000: loss=1.5775e-01
Epoch 70000: loss=1.5028e-01
Epoch 80000: loss=1.4436e-01
Epoch 90000: loss=1.3956e-01
Epoch 100000: loss=1.3558e-01
Epoch 110000: loss=1.3222e-01
Epoch 120000: loss=1.2935e-01
Epoch 130000: loss=1.2686e-01
Epoch 140000: loss=1.2469e-01
Epoch 150000: loss=1.2276e-01
Epoch 160000: loss=1.2105e-01
Epoch 170000: loss=1.1952e-01
Epoch 180000: loss=1.1813e-01
Epoch 190000: loss=1.1688e-01
Epoch 200000: loss=1.1573e-01
'''
# Let's see the result!
# y와 y_hat을 비교하여 정확도 계산
correct_cnt = (y == (y_hat > .5)).sum()
total_cnt = float(y.size(0))

print('Accuracy: %.4f' % (correct_cnt / total_cnt))
# Accuracy: 0.9666

df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach().numpy(),
                  columns=["y", "y_hat"])

sns.histplot(df, x='y_hat', hue='y', bins=50, stat='probability')
plt.show()