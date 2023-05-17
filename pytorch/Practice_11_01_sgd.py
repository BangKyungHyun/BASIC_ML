import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import  StandardScaler

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
imoprt torch.optim as optim

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

for i range(n_epochs):
    #suffle thn index to feed-forward
    indices = torch.randperm(x.size(0))
    x_ = torch.index_select(x, dim=0, index=indices)
    y_ = torch.index_select(y, dim=0, index=indices)

    x_ = x_.split(batch_size, dim=0)
    y_ = y_.split(batch_size, dim=0)

