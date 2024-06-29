#3장
#3.1.1 K-최근접 이웃


################################################################################
# 라이브러리 호출 및 데이터 준비
################################################################################
# 벡터 및 행렬의 연산 처리를 위한 라이브러리
import numpy as np
# 데이터를 차트나 플롯으로 그려주는 라이브러리
import matplotlib.pyplot as plt
# 데이터 분석 및 조작을 위한 라이브러리
import pandas as pd
# 모델 성능 평가
from sklearn import metrics

# 데이터셋에 열 이름 할당
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# 데이터를 판다스 데이터프레임에 저장
dataset = pd.read_csv('data/iris.data', names=names)

X = dataset.iloc[:, :-1].values # 모든 행을 사용하지만 열은 뒤에서 하나를 뺀 값을 사용
y = dataset.iloc[:, 4].values   # 모든 행을 사용하지만 열은 다섯번째만 값만 사용

print('X.shape =', X.shape)
print('y.shape =', y.shape)

# X.shape = (150, 4)
# y.shape = (150,)

################################################################################
# 훈련과 테스트 데이터셋 분리
################################################################################

from sklearn.model_selection import train_test_split

# X,Y을 사용하여 훈련과 테스트 데이터셋으로 분리하며, 테스트 데이터셋의 비율은 20% 만 사용
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
# 특성 스케일링 평균이 0, 표준편차가 1이 되도록 변환
s = StandardScaler()

# 훈련/테스트 데이터를 스케일링 처리
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)

################################################################################
# 모델 생성 및 훈련
################################################################################

from sklearn.neighbors import KNeighborsClassifier

# K=50인 K nearest neighbor 모델 생성
knn = KNeighborsClassifier(n_neighbors=50)
# 모델 훈련
knn.fit(X_train, y_train)

################################################################################
# 모델 정확도
################################################################################

from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print("정확도: {}".format( accuracy_score(y_test, y_pred)))

# 정확도: 0.9666666666666667
################################################################################
# 최적의 K 찾기
################################################################################
k=10
acc_array=np.zeros(k)
for k in np.arange(1,k+1,1):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1]=acc

max_acc=np.amax(acc_array)
acc_list=list(acc_array)
k=acc_list.index(max_acc)
print("정확도 ", max_acc, "으로 최적의 k는", k+1, "입니다.")

# 정확도  1.0 으로 최적의 k는 4 입니다.