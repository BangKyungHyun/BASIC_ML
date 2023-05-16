
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
df["TARGET"] = boston.target
df.tail()

scalar = StandardScaler()
scalar.fit(df.values[:, :-1])
df.values[:, :-1] = scalar.transform(df.values[:, :-1]).round(4)
df.tail()

# Train Model with PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        now         = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(format(step,',d'),'Epoch %d: loss=%.4e' % (i + 1, loss))

# Let's see the result!













