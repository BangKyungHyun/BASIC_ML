import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.DESCR)

df = pd.DataFrame(cancer.data, columns=cancer, feature_names)
df['class'] = cancer.target
'''
from matplotlib import pyplot as plt

x = torch.sort(torch.randn(100) * 10)[0]

print('x =', x )
'''