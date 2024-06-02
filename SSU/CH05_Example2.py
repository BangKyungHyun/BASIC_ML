import numpy as np

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

loaded_data = np.loadtxt('./CH05_data-01.csv',delimiter=',',dtype=np.float32)

x_data = loaded_data[ : , 0:-1]
t_data = loaded_data[ : , [-1]]

#데이터 차원 및 shape 확인
print("x_data.ndim = ", x_data.ndim, "x_data.shape =", x_data.shape )
print("t_data.ndim = ", t_data.ndim, "t_data.shape =", t_data.shape )

W = np.random.rand(3,1)
b = np.random.rand(1)
print("W =", W, "W.shape =", W.shape,"b =", b, "b.shape =", b.shape )

# loss function - 평균제곱오차(MSE) 계산
def loss_func(x,t):
    y = np.dot(x,W) + b
    #print("y = np.dot(x,W) + b = ", y, "t =", t,"(len(x) =", (len(x)))
    return ( np.sum((t-y)**2)) / (len(x))

# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 x : numpy type
def predict(x):
    y = np.dot(x,W) + b
    print("predict W =", W, "b =",b, "predict y =", y)
    return y

learning_rate = 1e-5

f = lambda x : loss_func(x_data,t_data )
print("Initial loss value = ", loss_func(x_data,t_data) )

for step in range(3000001):
    W -= learning_rate * derivative(f,W)
    b -= learning_rate * derivative(f,b)

    if (step % 100000 == 0):
        print("step =", step,"loss_func = ",loss_func(x_data,t_data), "W =",W,'b=',b)

test_data = np.array([100,98,81])
predict(test_data)
