import numpy as np
from datetime import datetime

def derivative(f, var):
    if var.ndim == 1:  # 1) b는 vector 이므로 1차원

        # print("A-1.var.ndim ==", var.ndim)
        #print("A-2.var ==", var)
        temp_var = var  #2) 원본 값 저장
        #print("A-3.temp_var ==",temp_var)

        delta = 1e-5
        # 3) 미분계수 보관 변수 초기화
        diff_val = np.zeros(var.shape   )
        #print("A-4.diff_val ==",diff_val)

        # 4) 벡터의 모든 열(column) 순서대로 반복함
        for index in range(len(var)):
            #print("A-5.index ==", index)
            #print("A-6.len(var) ==", len(var))

            # 원본값에서 index 순서대로 할당
            target_var = float(temp_var[index])
            #print("A-7.float(temp_var[index]) ==", float(temp_var[index]))
            #print("A-8.target_var ==", target_var)

            temp_var[index] = target_var + delta
            #print("A-9.temp_var[index] ==", temp_var[index])

            # x+delta 에 대한 평균제곱오차(MSE) 계산
            func_val_plus_delta = f(temp_var)
            #print("A-10.func_val_plus_delta =",func_val_plus_delta)

            temp_var[index] = target_var - delta
            #print("A-11.temp_var[index] ==", temp_var[index])

            # x-delta 에 대한 평균제곱오차(MSE) 계산
            func_val_minus_delta = f(temp_var)
            #print("A-12.func_val_minus_delta ==", func_val_minus_delta)

            # 미분계수 계산 (도함수)
            diff_val[index] = (func_val_plus_delta - func_val_minus_delta) / \
                              (2 * delta)

            #print("A-13.diff_val[index] ==", diff_val[index])

            # temp_var[index] 값이 delta에 의해 변경되어 변경전 값(target_var)을 할당
            temp_var[index] = target_var
            #print("A-14.temp_var[index] ==", temp_var[index])

        return diff_val

    elif var.ndim == 2:  # matrix
        #print("B-1.var.ndim ==", var.ndim)

        #print("B-2.var ==", var)
        temp_var = var
        #print("B-3.temp_var ==",temp_var)

        delta = 1e-5
        diff_val = np.zeros(var.shape)
        #print("B-4.diff_val ==",diff_val)

        rows = var.shape[0]
        columns = var.shape[1]

        #print("B-5.rows=", rows)
        #print("B-6.columns =", columns)

        for row in range(rows):
            #print("B-61.row ==", row)

            for column in range(columns):
                #print("B-62.column ==", column)

                target_var = float(temp_var[row, column])
                #print("B-7.float(temp_var[row, column]) ==", float(temp_var[row,column]))
                #print("B-8.target_var ==", target_var)

                temp_var[row, column] = target_var + delta
                #print("B-9.temp_var[row, column] ==", temp_var[row, column])

                # x+delta 에 대한 평균제곱오차(MSE) 계산
                func_val_plus_delta = f(temp_var)
                #print("B-10.func_val_plus_delta =", func_val_plus_delta)

                temp_var[row, column] = target_var - delta
                #print("B-11.temp_var[row, column] ==", temp_var[row, column])

                # x-delta 에 대한 평균제곱오차(MSE) 계산
                func_val_minus_delta = f(temp_var)
                #print("B-12.func_val_minus_delta =", func_val_minus_delta)

                # 미분계수 계산 (도함수)
                diff_val[row, column] = \
                    (func_val_plus_delta - func_val_minus_delta) / (2 * delta)
                #print("B-13.diff_val[row, column] ==", diff_val[row, column])

                # temp_var[row, coluemn] 값이 delta에 의해 변경되어 변경전 값(target_var)을 할당
                temp_var[row, column] = target_var
                #print("B-14.temp_var[row, column] ==", temp_var[row, column])

        return diff_val

def sigmoid(z):
    return 1 / (1+np.exp(-z))

x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10, 1)
t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10, 1)
print("x_data.shape = ", x_data.shape, ", t_data.shape = ", t_data.shape)

W = np.random.rand(1,1)
b = np.random.rand(1)

print("W = ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

# 최종출력은 y = sigmoid(Wx+b) 이며, 손실함수는 cross-entropy 로 나타냄
# loss function - cross-entropy 계산
def loss_func(x, t):

    delta = 1e-7  # log 무한대 발산 방지

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # cross-entropy (test data와 입력값에 가중치을 곱하고 bias을 합산값에 sigmoid 처리한 y와의 cross-entropy 계산
    #print("loss function  x =", x, "W =", W, "b =", b, "z = ", z, "sigmoid(z) =", y, \
    #"cross entropy = ", -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta)) )
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))

# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 test_data : numpy type
def predict(test_data):

    z = np.dot(test_data, W) + b
    y = sigmoid(z)

    if y >= 0.5:
        result = 1  # True
    else:
        result = 0  # False

    print("test_data =",test_data, "W =", W, "b =", b, "z =", z, "y=", y, "result =",result )
    return y, result

learning_rate = 1e-2  # 발산하는 경우, 1e-3 ~ 1e-6 등으로 바꾸어서 실행

f = lambda x: loss_func(x_data, t_data)

print("Initial loss value = ", loss_func(x_data, t_data))

start_time = datetime.now()

for step in range(50001):

    W -= learning_rate * derivative(f, W)
    b -= learning_rate * derivative(f, b)

    if (step % 5000 == 0):
        print("step = ", step, "loss value = ", loss_func(x_data, t_data))

end_time = datetime.now()

print("")
print("Elapsed Time => ", end_time - start_time)

test_data = np.array([3.0])
predict(test_data)

test_data = np.array([3.5])
predict(test_data)

test_data = np.array([11.0])
predict(test_data)

test_data = np.array([13.0])
predict(test_data)

test_data = np.array([17.0])
predict(test_data)

test_data = np.array([31.0])
predict(test_data)

