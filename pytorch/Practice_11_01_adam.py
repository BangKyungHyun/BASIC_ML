################################################################################
# 하이퍼 파라미터
# 하이퍼 파라미터란 모델의 성능에 영향을 주지만 데이터를 통해 자동으로 학습할 수 없는 파라미터를 가리킨다.
# 학습률, 신경망의 깊이/너비, 활성함수의 종류 등

# 적응형 학습률
# 학습률의 설정에 따라 모델의 학습 경향이 매우 달라질 수 있다.
# 학습 초반에는 큰 학습률이 선호되고, 학습 후반에는 작은 학습률이 선호된다.
# 이를 응용하여 각 가중치 파라미터별 학습 진행 정도에 따라 학습률을 다르게 자동 적용할 수 있다.
################################################################################
# Adam 최적화 방법
# 가장 널리 쓰이는 알고리즘
# 모멘텀과 적응형 학습률이 복합 적용된 방식
# 학습률 하이퍼 파라미터가 존재하지만, 입문 단계에서는 딱히 튜닝할 필요가 없다.
################################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()

df = pd.DataFrame(california.data, columns=california.feature_names)
df["Target"] = california.target
print('df.tail =', df.tail)

scalar = StandardScaler()
scalar.fit(df.values[:,:-1])
df.values[:,:-1] = scalar.transform(df.values[:,:-1])
print('df.tail =', df.tail)

# Train Model with PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df.values).float()
print('data.shape =', data.shape)

x = data[:, :-1]
y = data[:, -1:]
print('x.shape, y.shape =',x.shape, y.shape)

n_epochs = 1000
batch_size = 256
print_interval = 100

print('\nLearning hyper parameter => Epoch = ', format(n_epochs,','),'print_interval = ',format(print_interval,','))

#Build Model
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
#   (0): Linear(in_features=8, out_features=10, bias=True)
#   (1): LeakyReLU(negative_slope=0.01)
#   (2): Linear(in_features=10, out_features=9, bias=True)
#   (3): LeakyReLU(negative_slope=0.01)
#   (4): Linear(in_features=9, out_features=8, bias=True)
#   (5): LeakyReLU(negative_slope=0.01)
#   (6): Linear(in_features=8, out_features=7, bias=True)
#   (7): LeakyReLU(negative_slope=0.01)
#   (8): Linear(in_features=7, out_features=6, bias=True)
#   (9): LeakyReLU(negative_slope=0.01)
#   (10): Linear(in_features=6, out_features=1, bias=True)
# )
# 위 모델 사용시 cost(loss) value
# start time = 2023-05-31 04:47:57
# 2023-05-31 04:48:26 Epoch =  200  loss= 2.8430e-01
# 2023-05-31 04:48:56 Epoch =  400  loss= 2.7760e-01
# 2023-05-31 04:49:28 Epoch =  600  loss= 2.6646e-01
# 2023-05-31 04:50:02 Epoch =  800  loss= 2.6052e-01
# 2023-05-31 04:50:36 Epoch =  1,000  loss= 2.5804e-01
# 2023-05-31 04:51:12 Epoch =  1,200  loss= 2.5250e-01
# 2023-05-31 04:51:47 Epoch =  1,400  loss= 2.4967e-01
# 2023-05-31 04:52:20 Epoch =  1,600  loss= 2.4607e-01
# 2023-05-31 04:52:53 Epoch =  1,800  loss= 2.4620e-01
# 2023-05-31 04:53:27 Epoch =  2,000  loss= 2.4542e-01
# 2023-05-31 04:54:00 Epoch =  2,200  loss= 2.4289e-01
# 2023-05-31 04:54:35 Epoch =  2,400  loss= 2.4464e-01
# 2023-05-31 04:55:08 Epoch =  2,600  loss= 2.4167e-01
# 2023-05-31 04:55:41 Epoch =  2,800  loss= 2.3865e-01
# 2023-05-31 04:56:15 Epoch =  3,000  loss= 2.3928e-01
# 2023-05-31 04:56:49 Epoch =  3,200  loss= 2.3667e-01
# 2023-05-31 04:57:22 Epoch =  3,400  loss= 2.3696e-01
# 2023-05-31 04:57:55 Epoch =  3,600  loss= 2.3757e-01
# 2023-05-31 04:58:29 Epoch =  3,800  loss= 2.3426e-01
# 2023-05-31 04:59:02 Epoch =  4,000  loss= 2.3330e-01
# end time = 2023-05-31 04:59:02


################################################################################
# We don't need learning rate hyper-parameter
################################################################################
optimizer = optim.Adam(model.parameters())

now = datetime.datetime.now()
time1 = now.strftime('%Y-%m-%d %H:%M:%S')
print('start time =',time1)

for i in range(n_epochs):
    #Suffle the index to feed-forward
    indices = torch.randperm(x.size(0))
    x_ = torch.index_select(x, dim=0, index=indices)
    y_ = torch.index_select(y, dim=0, index=indices)

    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)
    # |x_[i]| = = (total_size, input_dim)
    # |y_[i]| = = (total_size, output_dim)
    y_hat = []
    total_loss = 0

    for x_i, y_i in zip(x_, y_):
        # |x_i| = |x_[i]|
        # |y_i| = |y_[i]|
        y_hat_i = model(x_i)
        loss = F.mse_loss(y_hat_i, y_i)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += float(loss)
        y_hat += [y_hat_i]

    total_loss = total_loss / len(x_)

    if (i + 1) % print_interval == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime, 'Epoch = ', format(i + 1, ','), ' loss=', '%.4e' % total_loss)

# 아래 모델 사용시 cost(loss) value
# model = nn.Sequential(
#     nn.Linear(x.size(-1), 6),
#     nn.LeakyReLU(),
#     nn.Linear(6, 5),
#     nn.LeakyReLU(),
#     nn.Linear(5, 4),
#     nn.LeakyReLU(),
#     nn.Linear(4, 3),
#     nn.LeakyReLU(),
#     nn.Linear(3, y.size(-1)),
# )
# 2023-05-30 17:46:27 Epoch 200: loss=3.4402e-01
# 2023-05-30 17:47:01 Epoch 400: loss=3.4021e-01
# 2023-05-30 17:47:33 Epoch 600: loss=3.3977e-01
# 2023-05-30 17:48:07 Epoch 800: loss=3.3885e-01
# 2023-05-30 17:48:54 Epoch 1000: loss=3.3830e-01
# 2023-05-30 17:49:47 Epoch 1200: loss=3.3887e-01
# 2023-05-30 17:50:35 Epoch 1400: loss=3.3832e-01
# 2023-05-30 17:51:09 Epoch 1600: loss=3.3888e-01
# 2023-05-30 17:51:46 Epoch 1800: loss=3.3059e-01
# 2023-05-30 17:52:33 Epoch 2000: loss=3.2038e-01

now = datetime.datetime.now()
time2 = now.strftime('%Y-%m-%d %H:%M:%S')
print('end time =',time2)

y_hat = torch.cat(y_hat, dim=0)
y = torch.cat(y_, dim=0)
# |y_hat| = (total_size, output_dim)
# |y| = (total_size, output_dim)

df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach().numpy(),
                  columns=["y", "y_hat"])

sns.pairplot(df, height=5)
plt.show()