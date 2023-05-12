import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 위스콘신 유방암 데이터셋은 30개의 속성을 가지며 이를 통해 유방방 여부를 예측
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.DESCR)

# 판다스로 데이터를 변환
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target

# 각 10개 속성이 평균, 표준편차, 최악값을 나타내고 있기 때문에 속성이 30개임
# 평균과 표준편차, 최악 속성들만 따로 모아서 class속성과 비교하는 페어플롯 그림
sns.pairplot(df[['class'] + list(df.columns[:10])])
plt.show()

'''
from matplotlib import pyplot as plt

x = torch.sort(torch.randn(100) * 10)[0]

print('x =', x )
'''