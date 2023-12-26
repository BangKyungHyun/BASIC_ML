################################################################################
# 라이브러리 호출
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN  # 밀도 기반 군집 분석
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA # 데이터 차원 축소

################################################################################
# 데이터 불러오기
################################################################################

X = pd.read_csv('data/credit_card.csv')
X = X.drop('CUST_ID', axis = 1)   # 불러온 데이터에서 'CUST_ID' 컬럼을 삭제
X.fillna(method ='ffill', inplace = True) # 결측값을 앞의 값으로 채움
print(X.head())

#        BALANCE  BALANCE_FREQUENCY  ...  PRC_FULL_PAYMENT  TENURE
# 0    40.900749           0.818182  ...          0.000000      12
# 1  3202.467416           0.909091  ...          0.222222      12
# 2  2495.148862           1.000000  ...          0.000000      12
# 3  1666.670542           0.636364  ...          0.000000      12
# 4   817.714335           1.000000  ...          0.000000      12

################################################################################
# 데이터 전 처리 및 데이터를 2차원으로 축소
################################################################################

scaler = StandardScaler()

# 평균이 0, 표준편차가 1이 되도록 데이터 크기를 조정
X_scaled = scaler.fit_transform(X)

# 데이터가 가우스 분포를 따르도록 정규화
X_normalized = normalize(X_scaled)

# 넘파이 배열을 데이터프레임으로 변환
X_normalized = pd.DataFrame(X_normalized)

# 2차원 차원 축소 선언
pca = PCA(n_components = 2)

# 차원 축소 적용
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head())
#
#          P1        P2
# 0 -0.489949 -0.679976
# 1 -0.519099  0.544829
# 2  0.330633  0.268876
# 3 -0.481657 -0.097609
# 4 -0.563512 -0.482506

################################################################################
# DBSCAN 모델 생성 및 결과의 시각화
################################################################################
# 모델 생성 및 훈련
db_default = DBSCAN(eps = 0.0375, min_samples = 3).fit(X_principal)
# 각 데이터 포인터에 할당된 모든 클러스터 레이블의 넘파일 배열을 labels에 저장
labels = db_default.labels_

# 출력 그래프의 색상을 위한 레이블 생성
colours = {}
colours[0] = 'y'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'

# 각 데이터 포인터에 대한 색상 벡터 생성
cvec = [colours[label] for label in labels]

# 플롯의 범례 구성
r = plt.scatter(X_principal['P1'], X_principal['P2'], color='y');
g = plt.scatter(X_principal['P1'], X_principal['P2'], color='g');
b = plt.scatter(X_principal['P1'], X_principal['P2'], color='b');
k = plt.scatter(X_principal['P1'], X_principal['P2'], color='k');

# 정의된 색상 벡터에 따라 x축에 p1, y축에 P2 플로팅
plt.figure(figsize=(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)

# 범례 구축
plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))
plt.show()

################################################################################
# 모델 튜닝
################################################################################

# 밀도 기반 군집 분석에서 사용하는 min_samples의 하이퍼파라미터를 3에서 50으로 변경
db = DBSCAN(eps=0.0375, min_samples=50).fit(X_principal)
labels1 = db.labels_

colours1 = {}
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'

cvec = [colours1[label] for label in labels1]
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

r = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[0])
g = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[1])
b = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[2])
c = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[3])
y = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[4])
m = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[5])
k = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[6])

plt.figure(figsize=(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.legend((r, g, b, c, y, m, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5',
            'Label -1'),
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8)
plt.show()

################################################################################
# 모델 튜닝
################################################################################

# 밀도 기반 군집 분석에서 사용하는 min_samples의 하이퍼파라미터를 50에서 100으로 변경
db = DBSCAN(eps=0.0375, min_samples=100).fit(X_principal)
labels1 = db.labels_

colours1 = {}
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'

cvec = [colours1[label] for label in labels1]
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

r = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[0])
g = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[1])
b = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[2])
c = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[3])
y = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[4])
m = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[5])
k = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[6])

plt.figure(figsize=(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.legend((r, g, b, c, y, m, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5',
            'Label -1'),
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8)
plt.show()