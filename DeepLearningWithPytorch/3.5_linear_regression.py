#3.1.4 선형 회귀

################################################################################
# 라이브러리 호출 및 데이터 준비
################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# %matplotlib inline

dataset = pd.read_csv('data/weather.csv')

################################################################################
# 데이터간 관계를 시각화로 표현
################################################################################

dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

################################################################################
# 데이터를 독립변수와 종속 변수로 분리하고 선형 회귀 모델 생성
################################################################################

# MinTemp에 따라 MaxTemp를 예측하기 위해, x변수는 MinTemp로 구성하고 y변수는 MaxTemp로 구성
X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()   # 선형 회귀 클래스를 사용
regressor.fit(X_train, y_train)  # fit()메서드를 사용하여 모델 훈현

################################################################################
# 회귀 모델에 대한 예측
################################################################################
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print('df =', df)

# df =     Actual  Predicted
# 0     25.2  23.413030
# 1     11.5  13.086857
# 2     21.1  27.264856
# 3     22.2  25.461874
# 4     20.4  26.937041
# ..     ...        ...
# 69    18.9  20.216833
# 70    22.8  27.674625
# 71    16.1  21.446140
# 72    25.1  24.970151
# 73    12.2  14.070302
################################################################################
# 테스트 데이터셋을 사용한 회귀선 표현
################################################################################

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

################################################################################
# 선형 회귀 모델 평가
################################################################################
print('평균제곱법:', metrics.mean_squared_error(y_test, y_pred))
print('루트 평균제곱법:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 평균제곱법: 17.011877668640622
# 루트 평균제곱법: 4.124545753006096




