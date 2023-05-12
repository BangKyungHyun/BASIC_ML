import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset from sklearn
# 위스콘신 유방암 데이터셋은 30개의 속성을 가지며 이를 통해 유방방 여부 예측
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.DESCR)

# 판다스로 데이터를 변환
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target

# Pair plot with mean features
# 각 10개 속성이 평균, 표준편차, 최악값을 나타내고 있기 때문에 속성이 30개임
# 평균과 표준편차, 최악 속성들만 따로 모아서 class속성과 비교하는 페어플롯 표출
#sns.pairplot(df[['class'] + list(df.columns[:10])])
#plt.show()

# Pair plot with std features
#sns.pairplot(df[['class'] + list(df.columns[10:20])])
# plt.show()

# Pair plot with worst features
sns.pairplot(df[['class'] + list(df.columns[20:30])])

# select features
cols = ["mean radius", "mean texture",
        "mean smoothness", "mean compactness", "mean concave points",
        "worst radius", "worst texture",
        "worst smoothness", "worst compactness", "worst concave points",
        "class"]

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
x = data[:, :-1]
y = data[:, -1:]

print('x.shape, y.shape =',x.shape, y.shape)
# x.shape, y.shape = torch.Size([569, 10]) torch.Size([569, 1])

n_epochs = 200000
learning_rate = 1e-2
print_interval = 10000

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

model = MyModel(input_dim=x.size(-1),
                output_dim=y.size(-1))
crit = nn.BCELoss()

optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate)

for i in range(n_epochs):
    y_hat = model(x)
    loss = crit(y_hat, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if (i + 1) % print_interval == 0:
        print('Epoch %d: loss=%.4e' % (i + 1, loss))

# Let's see the result!
correct_cnt = (y == (y_hat > .5)).sum()
total_cnt = float(y.size(0))

print('Accuracy: %.4f' % (correct_cnt / total_cnt))

df= pd.DataFrame(torch.cat([y,y_hat], dim=1).detach().numpy(),
                 columns=["y","y_hat"])
sns.hisplot(df, x='y_hat', hue='y', bins=50, stat='probability')
plt.show()
