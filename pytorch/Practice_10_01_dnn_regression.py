
# Regression using Deep Neural Networks
# Load Dataset from sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_boston
boston = load_boston

df = pd.DataFrame(boston.data, columns=boston_names)
df["TARGET"] = boston.target
df.tail()

scalar = StandardScaler
scalar.fit(df.values[:, :-1])
df.values[:, -1] = scalar.transform(df.values[:, :-1]).round(4)
df.tail()

# Train Model with PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df.values).float()
print('data.shape = ',data.shape)









