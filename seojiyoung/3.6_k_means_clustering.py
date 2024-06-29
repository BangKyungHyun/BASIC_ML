#3.2.1 K=평균 군집화

################################################################################
# 라이브러리 호출
################################################################################
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

################################################################################
# 상품에 대한 연 지출 호출
################################################################################

data = pd.read_csv('data/sales_data.csv')
print(data.head())

#    Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
# 0        2       3  12669  9656     7561     214              2674        1338
# 1        2       3   7057  9810     9568    1762              3293        1776
# 2        2       3   6353  8808     7684    2405              3516        7844
# 3        1       3  13265  1196     4221    6404               507        1788
# 4        2       3  22615  5410     7198    3915              1777        5185

# 명목형 자료 - 범주 간에 순서 의미가 없는 자료(혈액형)
categorical_features = ['Channel', 'Region']
# 연속형 자료 - 값이 연속적인 자료 ( 키, 몸무게)
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

for col in categorical_features:
    # 명목형 데이터는 판다스의 get_dummies() 메서드를 사용하여 숫자(0과1)로 변환
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
print(data.head())

#    Fresh  Milk  Grocery  Frozen  ...  Channel_2  Region_1  Region_2  Region_3
# 0  12669  9656     7561     214  ...          1         0         0         1
# 1   7057  9810     9568    1762  ...          1         0         0         1
# 2   6353  8808     7684    2405  ...          1         0         0         1
# 3  13265  1196     4221    6404  ...          0         0         0         1
# 4  22615  5410     7198    3915  ...          1         0         0         1

################################################################################
# 데이터에 전 처리(스케일링 적용)
################################################################################
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)

################################################################################
# 적당한 k 값 추출
################################################################################

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Optimal k')
plt.show()