import numpy as np

# 행렬에 열과 행 추가

A = np.array([ [10, 20, 30], [40, 50, 60] ])

print(A.shape)

# A matrix에 행(row) 추가할 행렬. 1행 3열로 reshape
# 행을 추가하기 때문에 우선 열을 3열로 만들어야 함.
row_add = np.array([70, 80, 90]).reshape(1, 3)

# A matrix 에 열(column) 추가할 행렬. 2행 1열 로 생성
# 열을 추가하기 때문에 우선 행을 2행로 만들어야 함.
column_add = np.array([1000, 2000]).reshape(2, 1)
print(column_add.shape)

# numpy.concatenate 에서 axis = 0 행(row) 기준
# A 행렬에 row_add 행렬 추가
B = np.concatenate((A, row_add), axis=0)

print(B)
# numpy.concatenate 에서 axis = 1 열(column) 기준
# B 행렬에 column_add 행렬 추가
C = np.concatenate((A, column_add), axis=1)

print(C)

loaded_data = np.loadtxt('data-01.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[ :, 0:-1]
t_data = loaded_data[ :, [-1]]

# 데이터 차원 및 shape 확인
print("x_data.ndim = ", x_data.ndim, ", x_data.shape = ", x_data.shape)
print("t_data.ndim = ", t_data.ndim, ", t_data.shape = ", t_data.shape)

# 0 ~ 1 사이의 random number 발생

random_number1 = np.random.rand(3)
random_number2 = np.random.rand(1, 3)
random_number3 = np.random.rand(3, 1)

print("random_number1 ==", random_number1, ", random_number1.shape ==", random_number1.shape)
print("random_number2 ==", random_number2, ", random_number2.shape ==", random_number2.shape)
print("random_number3 ==", random_number3, ", random_number3.shape ==", random_number3.shape)

X = np.array([2, 4, 6, 8])

print("np.sum(X) ==", np.sum(X))
print("np.exp(X) ==", np.exp(X))
print("np.log(X) ==", np.log(X))

X = np.array([2, 4, 6, 8])

print("np.max(X) ==", np.max(X))
print("np.min(X) ==", np.min(X))
print("np.argmax(X) ==", np.argmax(X))
print("np.argmin(X) ==", np.argmin(X))

X = np.array([ [2, 4, 6], [1, 2, 3], [0, 5, 8] ])

print("np.max(X) ==", np.max(X, axis=0)) # axis=0, 열기준
print("np.min(X) ==", np.min(X, axis=0)) # axis=0, 열기준

print("np.max(X) ==", np.max(X, axis=1)) # axis=1, 행기준
print("np.min(X) ==", np.min(X, axis=1)) # axis=1, 행기준

print("np.argmax(X) ==", np.argmax(X, axis=0)) # axis=0, 열기준
print("np.argmin(X) ==", np.argmin(X, axis=0)) # axis=0, 열기준

print("np.argmax(X) ==", np.argmax(X, axis=1)) # axis=1, 행기준
print("np.argmin(X) ==", np.argmin(X, axis=1)) # axis=1, 행기준

import matplotlib.pyplot as plt
import numpy as np

# 주피터 노트북을 사용하는 경우 노트북 내부에 그림 표시
#%matplotlib inline

# x data,  y data 생성
x_data = np.random.rand(100)
y_data = np.random.rand(100)

plt.title('scatter plot')
plt.grid()
plt.scatter(x_data, y_data, color='b', marker='o')
plt.show()
