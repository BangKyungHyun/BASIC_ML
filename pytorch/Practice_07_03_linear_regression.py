#!pip install matplotlib seaborn pandas sklearn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 실습에서 사용할 데이터셋을 불러옴
# Create Pandas Dataframe
from sklearn.datasets import load_boston
import datetime
boston = load_boston()

# 보스턴 주택 가격 데이터셋을 대한 설명이 자세하게 출력됨
print(boston.DESCR)

################################################################################
# 데이터셋은 506개의 샘플을 가지고 있으며 13개의 속성들과 이에 타깃값(label)을 갖고 있음
# 간단한 탐험적 데이터 분석을 위해 판다스 데이터 프레임으로 변환 후에 데이터 일부를 확인함
################################################################################

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["TARGET"] = boston.target
df.tail()
print("df.tail()\n", df.tail())

################################################################################
# Use the `shape` property
################################################################################

print('df.shape = ', df.shape)
# df.shape =  (506, 14)

################################################################################
# use the `len()` function with the `index` property
################################################################################

print('len(df.index) = ', len(df.index))
# len(df.index) =  506

print('list(df.columns) = ', list(df.columns))
# list(df.columns) =  ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
# 'TAX', 'PTRATIO', 'B', 'LSTAT', 'TARGET']

################################################################################
# 각 속성의 분포와 속성 사이의 선형적 관계 유무를 파악하기 위해 페어플롯을 그림
# sns.pairplot(df)
# plt.show()
################################################################################

################################################################################
# Target 속성에 대응하는 맨 마지막을 줄을 살펴보면 일부 속성들이 Target 속성과
# 약간의 선형적 관계를 띄는 것을 볼 수 있음
# 선형적 관계를 띄는 것으로 보이는 일부 속성을 추려 내여 다시 페어플롯을 그림
# 그림의 맨 첫 줄이 target 속성과 대응하여 그린 것임
################################################################################

cols = ["TARGET", "INDUS", "RM", "LSTAT", "NOX", "DIS"]

df[cols].describe()

# sns.pairplot(df[cols])
# plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Numpy 데이터를 파이토치 실수형 텐서로 변환함
# cols = ["TARGET", "INDUS", "RM", "LSTAT", "NOX", "DIS"] 데이터에 한해 학습 진행
data = torch.from_numpy(df[cols].values).float()

print("data.shape = ", data.shape)
#data.shape =  torch.Size([506, 6])

# 데이터를 입력 x와 출력 y로 나눔
x = data[:, 1:]
y = data[:, :1]

print("data = ", data, "x= ", x, "y= ", y )
print("x.shape = ", x.shape, "y.shape = ", y.shape)
# x.shape =  torch.Size([506, 5]) y.shape =  torch.Size([506, 1])

# 학습에 필요한 설정값 정함
n_epochs = 400000
learning_rate = 1e-5
print_interval = 10000

################################################################################
# 모델을 생성. 텐서x의 마지막 차원의 크기를 선형 계층의 입력 크기로 주고,
# 텐서 y의 마지막 차원의 크기를 선형 계층의 출력 크기로 함
################################################################################

################################################################################
# nn.Linear
################################################################################
# nn.Linear는 파이토치에서 사용되는 선형 변환(linear transformation)을 수행하는 클래스로,
# Fully Connected Layer 또는 Dense Layer라고도 불립니다.
#
# nn.Linear 클래스의 생성자(__init__)에는 다음과 같은 인수가 있습니다:
#
# in_features (int): 입력 텐서의 크기. 입력 텐서의 차원(dimension) 또는 특성(feature)의 수
# out_features (int): 출력 텐서의 크기. 출력 텐서의 차원(dimension) 또는 특성(feature)의 수
# bias (bool, optional): 편향(bias)을 사용할지 여부를 지정합니다. 기본값은 True입니다.

# nn.Linear 클래스는 두 개의 행렬 가중치(weight)와 편향(bias)을 학습하며,
# 입력 텐서를 선형 변환하여 출력 텐서를 생성합니다.
# 선형 변환은 입력 텐서와 가중치 행렬의 행렬 곱을 계산하고, 편향을 더하는 연산으로 이루어집니다.
#
# nn.Linear 클래스의 예제 코드는 다음과 같습니다:
#
# import torch
# import torch.nn as nn
#
# # 입력 텐서의 크기가 10이고 출력 텐서의 크기가 20인 선형 변환을 수행하는 nn.Linear 모듈 생성
# linear = nn.Linear(10, 20)
#
# # 입력 텐서 생성 (크기가 10인 벡터)
# input_tensor = torch.randn(1, 10)
#
# # 선형 변환 수행 (입력 텐서를 출력 텐서로 변환)
# output_tensor = linear(input_tensor)
#
# print("Input Tensor Size: ", input_tensor.size())
# print("Output Tensor Size: ", output_tensor.size())
#
# Input Tensor Size:  torch.Size([1, 10])
# Output Tensor Size:  torch.Size([1, 20])
# 위의 예제에서는 입력 텐서의 크기가 10이고 출력 텐서의 크기가 20인 nn.Linear 모듈을 생성하고,
# 입력 텐서를 선형 변환하여 출력 텐서를 생성하는 예제입니다.
# 출력 텐서의 크기는 nn.Linear의 out_features 인수에 지정한 값인 20과 동일합니다.


# ###############################################################################
# # nn.Linear 실습
# ###############################################################################
#
# import torch
# import torch.nn as nn
#
# m = nn.Linear(5, 1)
#
# print('m =', m)
# # m = Linear(in_features=5, out_features=1, bias=True)
#
# input = torch.randn(4, 5)
# print('input =', input)
#
# # input = tensor([[ 1.1559, -0.9744, -0.4955, -0.0991, -0.7903],
# #         [ 0.6457,  0.0207,  0.2133,  0.8778, -0.0765],
# #         [ 0.0709,  1.4191, -0.2880,  0.0706,  1.7034],
# #         [-1.0156,  0.3434,  0.4634,  0.9138,  1.0603]])
#
# print('m(input) =', m(input))
# # m(input) = tensor([[-0.6443],
# #         [-0.1951],
# #         [ 0.8806],
# #         [ 0.8775]], grad_fn=<AddmmBackward0>)
#
# output = m(input)
# print('output.size() = ', output.size())
#
# # output.size() =  torch.Size([4, 1])
#
# ###############################################################################
# a.size(-1)마지막 차원을 나타냅니다.
# 예를 들어 x의 모양이 (10,20)인 경우 x.size(-1)는 두 번째 차원, 즉 20을 참조합니다.
# print("x.size(-1) = ", x.size(-1), "y.size(-1) = ", y.size(-1))
# ###############################################################################

# x.size(-1) =  5 y.size(-1) =  1

# 입력 텐서의 크기가 5이고 출력 텐서의 크기가 1인 nn.Linear 모듈을 생성하고,
# 입력 텐서를 선형 변환하여 출력 텐서를 생성
# 실제 y값이 아닌 y 크기를 참조하기 위해 y.size(-1) 사용

model = nn.Linear(x.size(-1), y.size(-1))

print('model =', model)
# model = Linear(in_features=5, out_features=1, bias=True)

# Instead of implement gradient equation,
# we can use <optim class> to update model parameters, automatically.
# 옵티마이져 생성. 파이토치에서 제공하는 옵티마이저 클래스를 통해 최적화 작업을 수행
# backward함수를 호출한 후 옵티마이저 객체에서 step 함수를 호출하면 경사하강을
# 1회 수행한다.

################################################################################
# SGD : 확률적 경사 하강법(Stochastic Gradient Descent)
################################################################################
# 가중치를 업데이트할 때 미분을 통해 기울기를 구한 다음 기울기가 낮은 쪽으로 업데이트하겠다는 뜻

# 확률적(Stochastic)은 전체를 한번에 계산하지 않고 확률적으로 일부 샘플을 구해서 조금씩 나눠서 계산하겠다는 뜻
# n=1이고 남은 이미지에서 훈련 샘플을 무작위로 선택하는 경우, 이 프로세스를 확률적 경사 하강법
# (SGD, stochastic gradient descent)이라고 하는데, 이는 구현하고 시각화하기는 쉽지만 훈련 속도가 느리고
# (업데이트 과정이 많을수록) ‘노이즈가 많다’. 보통 ‘미니 배치 확률적 경사 하강법(mini-batch stochastic
# gradient descent)’을 선호하는 경향이 있다.

# Loss Function을 계산할 때 전체 Train-Set을 사용하는 것을 Batch Gradient Descent라고 한다.
# 그러나 이렇게 계산하면 한번 step을 내딛을 때, 전체 데이터에 대해 Loss Function을 계산해야 하므로 너무 많은 계산양을
# 필요로 한다. 이를 방지하기 위해 보통은 Stochastic Gradient Desenct(SGD)라는 방법을 사용한다. 이 방법에서는
# Loss Function을 계산할 때, 전체 데이터(Batch) 대신 일부 데이터의 모음(Mini-Batch)를 사용하여 Loss Function을
# 계산한다. Batch Gradient Descent보다 다소 부정확할 수는 있지만, 계산 속도가 훨씬 빠르기 때문에 같은 시간에 더 많은
# step을 갈 수 있으며, 여러 번 반복할 경우 Batch 처리한 결과로 수렴한다. 또한 Batch Gradient Descent에서 빠질
# Local Minima에 빠지지 않고 더 좋은 방향으로 수렴할 가능성도 높다. 여기에 더해 SGD를 변형시킨 여러 알고리즘들을 활용하면
# 훨씬 좋은 성능을 낼 수 있고, 변형된 알고리즘으로 Naive Stochastic Gradient Descent 알고리즘이고, Momentum, NAG,
# Adagrad, AdaDelta, RMSprop 등이 있다.

# torch.optim.SGD(params,lr,momentum,dampening = 0,weight_decay = 0, nesterov = False
# - params : 파라미터 그룹을 정의하거나 최적화하기 위한 파라미터의 반복 기능, 즉 모델의 파라미터를 넣어주면 된다.
# - Ir : learning rate의 약자이다.
# - momentum : 기본 값이 0인 momentum factor이다.
# - weight_decay :  가중치 감소로 기본 값이 0이다.
# - dampening : momentum을 위한 dampening이다. 기본 값 0
# - nesterov : Nesterov momentum을 사용할지 말 지를 결정한다. 기본 값 0

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

################################################################################
# 정해진 epoch만큼 for 반복문을 통해 최적화를 수행
################################################################################

for i in range(n_epochs):

    y_hat = model(x)

    loss = F.mse_loss(y_hat, y)
    # optimizer.zero_grad() 는 반복 시에 전에 계산했던 기울기를 0 으로 초기화하는 함수이다.
    # 즉 최적화된 모든 torch의 기울기를 0으로 바꾼다.
    # 기울기를 초기화해야 새로운 가중치와 편차에 대해서 새로운 기울기를 구할 수 있기 때문이다.
    optimizer.zero_grad()
    # w와 b에 대한 기울기 계산
    loss.backward()
    # model.paramters()에서 리턴되는 변수들의 기울기에 학습률을 곱해서 빼준 뒤에 업데이트한다.
    optimizer.step()

    if (i + 1) % print_interval == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime,'Epoch %d: loss=%.6e' % (i + 1, loss))


################################################################################
# 결과 확인- 모델을 통과한 y_hat를 가져와서 실제 y와 비교하기 위한 페어 플롯을 그림
################################################################################

df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(),
                  columns=["y", "y_hat"])

################################################################################
# 왼쪽 위에 그려진 y의 분포와 오른쪽 아래에 그려진 y_hat의 분포가 약간은 다르게 나타난 것을
# 볼수 있음. 하지만 오른쪽 위에 그려진 y와 왼쪽 아래의 y_hat과의 비교에서는 대부분의 점들이
# 빨간색 점선 부근에 나타나 있는 것을 확인할 수 있음
################################################################################

sns.pairplot(df, height=5)
plt.show()