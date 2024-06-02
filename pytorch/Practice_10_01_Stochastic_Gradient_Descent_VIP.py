################################################################################
# SGD는 비복원 추출을 통해 일부 샘플을 뽑아 미니배치를 구성하고
# 피드포워딩 및 파라미터(가중치) 업데이트를 수행하는 방법
# 기존 전체 데이터셋을 활용하는 방식에 비해 파라미터(가중치) 업데이트를 효율적으로 수행할 수 있음
################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_california_housing
califonia = fetch_california_housing()

df = pd.DataFrame(califonia.data,columns=califonia.feature_names)
df["Target"] = califonia.target
print('df.tail() = \n',df.tail())

sns.pairplot(df.sample(1000))
# plt.show()
################################################################################
# 정규화
################################################################################
scalar = StandardScaler()
scalar.fit(df.values[:, :0-1])
df.values[:, :-1] = scalar.transform(df.values[:, :-1])

sns.pairplot(df.sample(1000))
# plt.show()

# Train Model with PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 정제된 데이터를 파이토치 텐서로 변환하고 크기를 확인
data = torch.from_numpy(df.values).float()
print('data.shape = ',data.shape)
# data.shape =  torch.Size([20640, 9])

# 입력 데이터와 출력 데이터를 분리하여 각각 x와 y에 저장
x = data[:, :-1]
y = data[:, -1:]
print('x.shape, y.shape = ',x.shape, y.shape)
# x.shape, y.shape =  torch.Size([20640, 8]) torch.Size([20640, 1])

# 학습에 필요한 하이퍼파라미터 설정
# 모덱은 전체 데이터셋의 모든 샘플을 천번 학습, 배치사이즈는 256, 학습률은 0.01로 지정
n_epochs = 10000
batch_size = 256
print_interval = 100
learning_rate = 1e-5

# print('\nLearning hyper parameter => Epoch = %d,: print_interval = %d \n' % (n_epochs, print_interval))
print('\nLearning hyper parameter => Epoch = ', format(n_epochs,','),'print_interval = ',format(print_interval,','))
################################################################################
# nn.Sequential 클래스를 활용하여 심층신경망을 구성.nn.Sequential을
# 선언할 때, 선형 계층 nn.Linear와 활성함수 nn.LeakyReLU를 선언
# 주의할 점
# 1) 선형계층과 마지막 선형 계층은 실제 데이터셋 텐서 x의 크기(8)와 y의 크기(1)를
#    입출력 크기로 갖도록 정함
# 2) 내부의 선형 계층들은 서로 입출력 크기가 호환 되도록 되어 있다는 점에도 주목
################################################################################
# Build models
model = nn.Sequential(
    nn.Linear(x.size(-1), 10),
    nn.LeakyReLU(),
    nn.Linear(10, 9),
    nn.LeakyReLU(),
    nn.Linear(9, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 7),
    nn.LeakyReLU(),
    nn.Linear(7, 6),
    nn.LeakyReLU(),
    nn.Linear(6, y.size(-1)),
)

print('model =', model)

# model = Sequential(
#   (0): Linear(in_features=8, out_features=6, bias=True)
#   (1): LeakyReLU(negative_slope=0.01)
#   (2): Linear(in_features=6, out_features=5, bias=True)
#   (3): LeakyReLU(negative_slope=0.01)
#   (4): Linear(in_features=5, out_features=4, bias=True)
#   (5): LeakyReLU(negative_slope=0.01)
#   (6): Linear(in_features=4, out_features=3, bias=True)
#   (7): LeakyReLU(negative_slope=0.01)
#   (8): Linear(in_features=3, out_features=1, bias=True)
# )

# 옵티마이저 정의
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
################################################################################
# |x| = = (total_size, input_dim)
# |y| = = (total_size, output_dim)

# 바깥쪽 for 반복문은 정해진 최대 에포크 수 만큼 반복을 수행하여, 모델이 데이터셋을
# n_epochs 만큼 반복해서 학습
#
# 안쪽 for 반복문은 미니배치에 대해서 피드포워딩(feed-forwarding)과 역전파
# (back-propagation), 그리고 경사하강(gradient descent)을 수행
#
# 안쪽 for 반복문 앞을 보면 매 에포크마다 데이터셋을 랜덤하게 섞어(셔플링shuffling)
# 주고 미니배치로 나눔
#
# 이때 중요한 점은 입력 텐서 x와 출력 텐서 y를 각각 따로 셔플링을 수행하는 것이 아니라,
# 함께 동일하게 섞음
#
# 만약 따로 섞어주게 된다면 x와 y의 관계가 끊어지게되어 아무 의미없는
# 노이즈로 가득찬 데이터가 됨
#
# 이 과정을 좀 더 자세히 살펴보면 randperm 함수를 통해서 새롭게 섞어줄 데이터셋의 인덱스
# 순서를 정함

# 그리고 index_select 함수를 통해서 이 임의의 순서로 섞인 인덱스 순서대로 데이터셋을
# 섞어 줌
# 마지막으로 split 함수를 활용하여 원하는 배치사이즈로 텐서를 나누어주면 미니배치를 만드는
# 작업이 끝남
#
# 안쪽 for 반복문은 전체 데이터셋 대신에 미니배치 데이터를 모델에 학습시킨다는 점이 다를 뿐,
# 앞서 챕터들에서 보았던 코드들과 동일
# 하나 추가된 점은 y_hat이라는 빈 리스트를 만들어, 미니배치마다 y_hat_i 변수에
# 피드포워딩 결과가 나오면 y_hat에 차례대로 저장

# 그리고 마지막 에포크가 끝나면 이 y_hat 리스트를 파이토치 cat 함수를 활용하여
# 이어 붙여 하나의 텐서로 만든 후, 실제 정답과 비교
################################################################################
now = datetime.datetime.now()
time1 = now.strftime('%Y-%m-%d %H:%M:%S')
print('start time =',time1)

for i in range(n_epochs):

    # suffle the index to feed-forward
    # randperm 함수를 통해서 새롭게 섞어줄 데이터셋의 인덱스 순서를 정함
    indices = torch.randperm(x.size(0))

    # index_select 함수를 통해서 이 임의의 순서로 섞인 인덱스 순서대로 데이터셋을 섞음
    x_ = torch.index_select(x, dim=0, index=indices)
    y_ = torch.index_select(y, dim=0, index=indices)

    # split 함수를 활용하여 원하는 배치사이즈로 텐서를 나누어 주면 미니배치를 만드는
    # 작업이 끝남
    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)
    # |x_[i]| = = (total_size, input_dim)
    # |y_[i]| = = (total_size, output_dim)

    y_hat = []
    total_loss = 0

    for x_i, y_i in zip(x_, y_):
        # |x_i| = |x_[i]|
        # |y_i| = |y_[i]|

        # 신경망 모델 결과를 y_hat_i에 할당
        y_hat_i = model(x_i)

        # 실제 target(label)과 학습결과간의 차이(비용, 손실)을 계산
        loss = F.mse_loss(y_hat_i, y_i)

        # optimizer.zero_grad() 는 반복 시에 전에 계산했던 기울기를 0 으로 초기화 하는 함수
        # 즉 최적화된 모든 torch의 기울기를 0으로 바꿈
        # 기울기를 초기화 해야 새로운 가중치와 편차에 대해서 새로운 기울기를 구할 수 있기 때문
        optimizer.zero_grad()

        # w와 b에 대한 기울기 계산
        loss.backward()

        # model.paramters()에서 리턴되는 변수들의 기울기에 학습률을 곱해서 빼준 뒤에 업데이트한다.
        optimizer.step()

        ################################################################################
        # loss 변수에 담긴 손실 값 텐서를 float type casting을 통해 단순 float 타입으로 변환하여
        # total_loss 변수에 더하는 것을 볼 수 있음.이 부분이 매우 중요함
        ################################################################################
        # 타입캐스팅 이전의 loss 변수는 파이토치 텐서 타입으로 그래디언트를 가지고 있음

        # 파이토치의 AutoGrad 동작 원리에 의해서 loss 변수가 계산될 때까지 활용된
        # 파이토치 텐서 변수들이 loss 변수에 줄줄이 엮여 있음

        # 따라서 만약 float 타입캐스팅이 없다면 total_loss도 파이토치 텐서가 될 것이고,
        # 이 total_loss 변수는 해당 에포크의 모든 loss 변수를 엮고 있음

        # 결과적으로 total_loss가 메모리에서 없어지지 않는다면 loss 변수와 그에 엮인 텐서 변수들 모두가
        # 아직 참조 중인 상태이므로 파이썬 garbage collector에 의해서 메모리에서 해제되지 않음
        # 즉, memory leak이 발생하게 됨

        # 더욱이 추후 실습에서처럼 손실 곡선을 그려보기 위해서 total_loss 변수를 따로 또 저장하기라도
        # 한다면 학습이 끝날때까지 학습에 사용된 대부분의 파이토치 텐서 변수들이 메모리에서 해제되지 않는
        # 최악의 상황이 발생할 수도 있음.

        # 그러므로 앞서와 같은 상황에서는 float 타입캐스팅 또는 detach 함수를 통해 AutoGrad를 위해
        # 연결된 그래프를 잘라내는 작업이 필요

        total_loss += float(loss) #This is very important to prevent memory leak

        # 미니배치마다 y_hat_i 변수에 피드포워딩 결과가 나오면 y_hat에 차례대로 저장
        y_hat += [y_hat_i]

        total_loss = total_loss / len(x_)

    if( i + 1 ) % print_interval == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime, 'Epoch = ', format(i + 1,',') , ' loss=','%.4e'%total_loss)

now = datetime.datetime.now()
time2 = now.strftime('%Y-%m-%d %H:%M:%S')
print('end time =',time2)

# 마지막 에포크가 끝나면 이 y_hat 리스트를 파이토치 cat 함수를 활용하여 이어붙여
# 하나의 텐서로 만든 후, 실제 정답과 비교
y_hat = torch.cat(y_hat, dim=0)
y = torch.cat(y_, dim=0)
# |y_hat| = (total_size, output_dim)
# |y| = (total_size, output_dim)

# Let's see the result!
df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach().numpy(),columns=["y", "y_hat"])

# 페어플랏을 통해 확인해보면, 조금 넓게 퍼져있긴 하지만, 대체로 중앙을 통과하는
# 대각선 주변으로 점들이 분포하고 있는 것을 볼 수 있음
sns.pairplot(df, height=5)
plt.show()