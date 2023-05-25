import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_california_housing
califonia = fetch_california_housing()

df = pd.DataFrame(califonia.data,columns=califonia.feature_names)
df["Target"] = califonia.target
print('df.tail() = \n',df.tail())

sns.pairplot(df.sample(1000))
plt.show()

scalar = StandardScaler()
scalar.fit(df.values[:, :0-1])
df.values[:, :-1] = scalar.transform(df.values[:, :-1])

sns.pairplot(df.sample(1000))
plt.show()

# Train Model with PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df.values).float()
print('data.shape = ',data.shape)

x = data[:, :-1]
y = data[:, -1:]
print('x.shape, y.shape = ',x.shape, y.shape)

n_epochs = 4000
batch_size = 256
print_interval = 200
learning_rate = 1e-2

# Build models
model = nn.Sequential(
    nn.Linear(x.size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 4),
    nn.LeakyReLU(),
    nn.Linear(4, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y.size(-1)),
)
print('model =', model)

optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# |x| = = (total_size, input_dim)
# |y| = = (total_size, output_dim)

for i in range(n_epochs):
    #suffle thn index to feed-forward
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

        total_loss += float(loss) # This is very important to prevent memory leak
        y_hat += [y_hat_i]

    total_loss = total_loss / len(x_)
    if( i + 1 ) % print_interval == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime, 'Epoch %d: loss=%.4e' % (i + 1, total_loss))

y_hat = torch.cat(y_hat, dim=0)
y = torch.cat(y_, dim=0)
# |y_hat| = (total_size, output_dim)
# |y| = (total_size, output_dim)



