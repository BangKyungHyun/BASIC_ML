# Regression using Deep Neural Networks
# Load Dataset from sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
# StandardScaler는 모든 피처들을 평균이 0, 분산이 1인 정규분포를 갖도록 만듬
from sklearn.preprocessing import StandardScaler
# scikit-learn dataset에서 boston dataset을 load
# scikit-learn dataset에서 sample data import
from sklearn.datasets import load_boston

################################################################################
# boston dataset load
################################################################################
boston = load_boston()
################################################################################
# 데이터 프레임으로 변환
################################################################################
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# 출력값은 TARGET 속성으로 저장되도록 함
df["TARGET"] = boston.target
print('df.tail() =\n',df.tail())
# df.tail() =
#          CRIM   ZN  INDUS  CHAS    NOX  ...    TAX  PTRATIO       B  LSTAT  TARGET
# 501  0.06263  0.0  11.93   0.0  0.573  ...  273.0     21.0  391.99   9.67    22.4
# 502  0.04527  0.0  11.93   0.0  0.573  ...  273.0     21.0  396.90   9.08    20.6
# 503  0.06076  0.0  11.93   0.0  0.573  ...  273.0     21.0  396.90   5.64    23.9
# 504  0.10959  0.0  11.93   0.0  0.573  ...  273.0     21.0  393.45   6.48    22.0
# 505  0.04741  0.0  11.93   0.0  0.573  ...  273.0     21.0  396.90   7.88    11.9

# [5 rows x 14 columns]


# missing value 검사
print('df.isnull().sum() = \n', df.isnull().sum())

# df.isnull().sum() =
#  CRIM       0
# ZN         0
# INDUS      0
# CHAS       0
# NOX        0
# RM         0
# AGE        0
# DIS        0
# RAD        0
# TAX        0
# PTRATIO    0
# B          0
# LSTAT      0
# TARGET     0
# dtype: int64

################################################################################
# 보스톤 주택 가격 데이터넷은 13개의 속성을 가지며 506개의 샘플로 구성되어 있음.
# 일부 속성만을 활용하여 선형 회귀를 학습했던 것과 달리 이번에는 전체 속성들을 활용하여
# 심층신경망 학습을 진행함
# 조금 더 쉽고 수월한 최적화 및 성능향샹을 위해 표준 스케줄링을 통해 입력값을 정규화 함
# 보스턴 주택 가격 데이터셋의 각 열이 정규분포를 따른다고 가정하고 표준 스케줄링을 적용함
# 다음 테이블은 표준 스케일링을 적용한 결과를 보여 줌
################################################################################

scalar = StandardScaler()
scalar.fit(df.values[:, :-1])
df.values[:, :-1] = scalar.transform(df.values[:, :-1]).round(4)

print('df.tail() =\n',df.tail())
# df.tail() =
#         CRIM      ZN   INDUS    CHAS  ...  PTRATIO       B   LSTAT  TARGET
# 501 -0.4132 -0.4877  0.1157 -0.2726  ...   1.1765  0.3872 -0.4181    22.4
# 502 -0.4152 -0.4877  0.1157 -0.2726  ...   1.1765  0.4411 -0.5008    20.6
# 503 -0.4134 -0.4877  0.1157 -0.2726  ...   1.1765  0.4411 -0.9830    23.9
# 504 -0.4078 -0.4877  0.1157 -0.2726  ...   1.1765  0.4032 -0.8653    22.0
# 505 -0.4150 -0.4877  0.1157 -0.2726  ...   1.1765  0.4411 -0.6691    11.9
#
# [5 rows x 14 columns]

# Train Model with PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

################################################################################
# 학습에 필요한 패키지를 불려오고 판다스에 저장된 넘파이 값을 파이토치 텐서로 변환하여
# 입력 텐서 x와 출력 텐서 y를 만듬
################################################################################
data = torch.from_numpy(df.values).float()
print('data.shape = ',data.shape)
# data.shape =  torch.Size([506, 14])

x = data[:, :-1]
y = data[:, -1:]
print('x.shape, y.shape =' , x.shape, y.shape)
# x.shape, y.shape = torch.Size([506, 13]) torch.Size([506, 1])

n_epochs = 1000000
learning_rate = 1e-4
print_interval = 10000

# Build Model using nn.Module
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(0.1)     # 새는, 구멍 난
##############################################################################
# 심층 신경망 정의
##############################################################################
# nn.Module을 상속받아 Mymodel 이라는 나만이 모델 클래스를 정의함
# 이러한 심층신경망은 4개의 선형 계층과 1개의 비선형을 함수를 갖도록 함
# __init__함수를 살펴보면 선형계층은 각각 linear1,linear2,linear3,linear4라는 이름을
# 가지도록 선언하고 비선형 함수는 ReLU을 사용.
#
# 다만 선형계층들은 각각 다른 가중치 파리미터를 가지게 되므로 다른 객체로 선언
#
# 비선형 활성 함수는 학습되는 파라미터를 갖지 않았기 때문에 모든 계층에서
# 동일하게 동작하므로 한 개만 선언하여 재활용함
#
# 각 선형 계층의 입출력 크기는 최초 입력 자원(input_dim)과 최종 출력 차원(output_dim)을
# 제외하고는 임의로 정함
#
# forward 함수에서는 앞서 선언된 내부 모듈들을 활용하여 피드포워드 연산을 수행할 수 있도록 함
#
# x라는 샘플 갯수 곱하기 입력 차원(batch_size, input_dim) 크기의 2차원 텐서가 주어지면
# 최종적으로 샘플 갯수 곱하기 출력 차원원(batch_size, output_dim)크기의 2차원 텐서로 뱉어
# 내는 함수가 됨
#
# 여기에서 input_dim과 output_dim은 __init__함수에서 미리 입력 받는 것을 볼 수 있음
##############################################################################
# 마지막 계층에서는 활성함수를 씌우지 않도록 주의해야 함
##############################################################################

class MyModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super().__init__()

        self.linear1 = nn.Linear(input_dim,3)
        self.linear2 = nn.Linear(3, 3)
        self.linear3 = nn.Linear(3, 3)
        self.linear4 = nn.Linear(3, output_dim)
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

# model 1 =   MyModel(
#   (linear1): Linear(in_features=13, out_features=3, bias=True)
#   (linear2): Linear(in_features=3, out_features=3, bias=True)
#   (linear3): Linear(in_features=3, out_features=3, bias=True)
#   (linear4): Linear(in_features=3, out_features=1, bias=True)
#   (act): ReLU()
# )
##############################################################################
# Build Model with LeakyReLU using nn.Sequential
##############################################################################
# 앞서와 같은 방법으로 직접 나만의 모델 클래스를 정의하는 방법도 아주 좋은 방법입니다.
# 하지만 지금은 모델 구조가 매우 단순한 편입니다. 입력 텐서를 받아 단순하게 순차적으로
# 앞으로 하나씩 계산해 나가는 것에 불과하기 때문입니다. 이 경우에는 나만의 모델 클래스를
# 정의하는 대신, 다음과 같이 nn.Sequential 클래스를 활용하여 훨씬 더 쉽게 모델 객체를
# 선언할 수 있습니다. 다음은 앞서 MyModel 클래스와 똑같은 구조를 갖는 심층신경망을
# nn.Sequential 클래스를 활용하여 정의하는 모습입니다. 단순히 내가 원하는 연산을 수행할
# 내부 모듈들을 nn.Sequential을 생성할 때, 순차적으로 넣어주는 것을 볼 수 있습니다.
# 당연한 것이지만 앞 쪽에 넣은 모듈들의 출력이 바로 뒷 모듈의 입력에 될 수 있도록,
# 내부 모듈들의 입출력 크기를 잘 적어주어야 할 것입니다.
##############################################################################

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

# model  2 =  Sequential(
#   (0): Linear(in_features=13, out_features=3, bias=True)
#   (1): LeakyReLU(negative_slope=0.01)
#   (2): Linear(in_features=3, out_features=3, bias=True)
#   (3): LeakyReLU(negative_slope=0.01)
#   (4): Linear(in_features=3, out_features=3, bias=True)
#   (5): LeakyReLU(negative_slope=0.01)
#   (6): Linear(in_features=3, out_features=3, bias=True)
#   (7): LeakyReLU(negative_slope=0.01)
#   (8): Linear(in_features=3, out_features=1, bias=True)
# )

optimizer = optim.SGD(model.parameters(),lr=learning_rate)

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

# 2023-05-19 09:50:25 Epoch 5000: loss=1.0746e+00
# 2023-05-19 09:50:32 Epoch 10000: loss=1.0616e+00
# 2023-05-19 09:50:39 Epoch 15000: loss=1.0522e+00
# 2023-05-19 09:50:45 Epoch 20000: loss=1.0445e+00
# 2023-05-19 09:50:51 Epoch 25000: loss=1.0379e+00
# 2023-05-19 09:50:57 Epoch 30000: loss=1.0324e+00
# 2023-05-19 09:51:04 Epoch 35000: loss=1.0276e+00
# 2023-05-19 09:51:10 Epoch 40000: loss=1.0236e+00
# 2023-05-19 09:51:17 Epoch 45000: loss=1.0202e+00
# 2023-05-19 09:51:27 Epoch 50000: loss=1.0172e+00
# 2023-05-19 09:51:37 Epoch 55000: loss=1.0147e+00
# 2023-05-19 09:51:45 Epoch 60000: loss=1.0125e+00
# 2023-05-19 09:51:53 Epoch 65000: loss=1.0105e+00
# 2023-05-19 09:52:00 Epoch 70000: loss=1.0088e+00
# 2023-05-19 09:52:07 Epoch 75000: loss=1.0074e+00
# 2023-05-19 09:52:14 Epoch 80000: loss=1.0060e+00
# 2023-05-19 09:52:24 Epoch 85000: loss=1.0048e+00
# 2023-05-19 09:52:31 Epoch 90000: loss=1.0038e+00
# 2023-05-19 09:52:38 Epoch 95000: loss=1.0028e+00
# 2023-05-19 09:52:45 Epoch 100000: loss=1.0019e+00

##############################################################################
# Let's see the result!  (샘플대로 했으나 오류 발생)
##############################################################################
df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach().numpy(),
                  columns=["y", "y_hat"])

sns.pairplot(df, height=5)
plt.show()
