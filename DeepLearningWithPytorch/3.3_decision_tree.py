#3.1.3 의사결정 트리

################################################################################
# 라이브러리 호출 및 데이터 준비
################################################################################
import pandas as pd

# 판다스를 이용하여 titanic_train.csv 파일을 로드해서 df에 저장
df = pd.read_csv('data/titanic_train.csv', index_col='PassengerId')
print(df.head())
#
#              Survived  Pclass  ... Cabin Embarked
# PassengerId                    ...
# 1                   0       3  ...   NaN        S
# 2                   1       1  ...   C85        C
# 3                   1       3  ...   NaN        S
# 4                   1       1  ...  C123        S
# 5                   0       3  ...   NaN        S
################################################################################
# 데이터 전처리
################################################################################

# 승객의 생존 여부를 예측하려고 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'사용
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# 성별을 나타내는 sex을 0 또는 1의 정수 값으로 표현
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna()     # 값이 없는 데이터는 삭제
X = df.drop('Survived', axis=1)
y = df['Survived']    # 'Survived' 값을 예측 레이블로 사용

################################################################################
# 훈련과 테스트 데이터셋으로 분리
################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

################################################################################
# 결정 트리 모델 생성
################################################################################

from sklearn import tree
model = tree.DecisionTreeClassifier()

################################################################################
# 모델 훈련
################################################################################

model.fit(X_train, y_train)

################################################################################
# 모델 예측
################################################################################

y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
print('accuracy_score(y_test, y_predict) = ',accuracy_score(y_test, y_predict))

# accuracy_score(y_test, y_predict) =  0.8324022346368715

################################################################################
# 혼동 모델을 이용한 성능 측정
################################################################################

from sklearn.metrics import confusion_matrix
print(pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
))

#                    Predicted Not Survival  Predicted Survival
# True Not Survival                      97                  15
# True Survival                          16                  51
pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
)