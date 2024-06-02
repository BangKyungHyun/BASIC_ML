import numpy as np

def derivative(f, var):
    if var.ndim == 1:  # 1) b는 vector 이므로 1차원
        #print("A-1.var.ndim ==", var.ndim)

        #print("A-2.var ==", var)
        temp_var = var  #2) 원본 값 저장
        #print("A-3.temp_var ==",temp_var)

        delta = 1e-5
        # 3) 미분계수 보관 변수 초기화
        diff_val = np.zeros(var.shape)
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

            # x+delta 에 대한 평균제곱오차(MSE) 구현
            func_val_plus_delta = f(temp_var)
            #print("A-10.func_val_plus_delta =",func_val_plus_delta)

            temp_var[index] = target_var - delta
            #print("A-11.temp_var[index] ==", temp_var[index])

            # x-delta 에 대한 평균제곱오차(MSE) 구현
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

                # x+delta 에 대한 평균제곱오차(MSE) 구현
                func_val_plus_delta = f(temp_var)
                #print("B-10.func_val_plus_delta =", func_val_plus_delta)

                temp_var[row, column] = target_var - delta
                #print("B-11.temp_var[row, column] ==", temp_var[row, column])

                # x-delta 에 대한 평균제곱오차(MSE) 구현
                func_val_minus_delta = f(temp_var)
                #print("B-12.func_val_minus_delta =", func_val_minus_delta)

                # 미분계수 계산 (도함수)
                diff_val[row, column] = \
                    (func_val_plus_delta - func_val_minus_delta) / (2 * delta)
                #print("B-13.diff_val[row, column] ==", diff_val[row, column])

                # temp_var[row, column] 값이 delta에 의해 변경되어 변경전 값(target_var)을 할당
                temp_var[row, column] = target_var
                #print("B-14.temp_var[row, column] ==", temp_var[row, colum#n])

        return diff_val

x_data = np.array([1,2,3,4,5]).reshape(5,1) #입력 데이터 할당
t_data = np.array([2,3,4,5,6]).reshape(5,1) #정답 데이터 할당
print("x_data.shape = ", x_data.shape, ", t_data.shape = ", t_data.shape)

W = np.random.rand(1,1) # 가중치 W 초기화
b = np.random.rand(1)   # 바이어스  b 초기화

# cost function - 평균제곱오차(Mean Square Error) 계산
def cost_func(x_data,t_data):

    y = np.dot(x_data,W) + b
    #print("cost func W =", W, "b =", b, "np.sum((t-y)**2)) =", np.sum((t_data-y)**2), \
    #"(len(x)) = ", (len(x_data)), "( np.sum((t-y)**2)) / (len(x)) =", ( np.sum((t_data-y)**2)) / (len(x_data)) )

    return ( np.sum((t_data-y)**2)) / (len(x_data))

# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함#수
# 입력변수 x : numpy type
def predict(x):
    print("predict W =", W, "b =",b)
    y = np.dot(x,W) + b
    print("predict y =", y)

    return y

learning_rate = 1e-2

# lambda 뒤에 명칭은 xxxxx 등으로 제약없이 사용 가능
f1 = lambda xxxxx: cost_func(x_data, t_data)

print("Initial cost_value =", cost_func(x_data, t_data), "Initial W = ",W ,"\n", "b =",b)

for step in range(10001):
    # W는 Matrix 자료형으로 derivateive 함수에서 2단계(var.ndim == 2)를 수행하고
    # b는 vector 자료형으로 derivateive 함수에서 1단계(var.ndim == 1)를 수행함

    ######################################################################
    # tensorflow을 이용하여 수동으로 W, b 계산
    # lab-03-2-minimizing_cost_gradient_update.py 참조
    # minimize: Gradient Descent using derivative: W -= learning_rate * derivative
    # 최소화: 미분함수를 사용한 경사하강법: W -= learning_rate * 미분함수
    # ---------------------------------------------------------------------
    # Our hypothesis for linear model X * W + b
    # hypothesis = X * W + b
    # ---------------------------------------------------------------------
    # cost/loss function
    # cost = tf.reduce_mean(tf.square(hypothesis - Y))
    # ---------------------------------------------------------------------
    # gradient = tf.reduce_mean((W * X - Y) * X)  # 기울기
    # descent = W - learning_rate * gradient  # 가중치에서 러닝레이트 * 경사각을 빼줌
    # update = W.assign(descent)
    ######################################################################

    ######################################################################
    # 한조각 교재 100페이지 참조
    # 선형 회귀의 목표는 트레이닝 데이터의 특성과 분표를 가장 잘 나타낼 수 있는 임의의
    # 직선 y = Wx+b에서의 가중치 W와 바이어스 b를 구하는 것이기 때문에 손실함수를
    # 식[5.3]과 같이 정의하였고 경사하강법을 이용하여 이러한 손실함수가 최솟값을 갖도록
    # 하는 가중치 W와 바이어스 b를 편미분으로 구한다는 것을 알 수 있다.
    ######################################################################
    # 미분함수를 이용하여 수동으로 W, b 계산
    #---------------------------------------------------------------------
    # W -= learning_rate * derivative(f1, W)
    # b -= learning_rate * derivative(f1, b)
    #---------------------------------------------------------------------
    # 1. 손실함수에 대한 미분계수 계산
    #---------------------------------------------------------------------
    #         df(x)      f(x + ^x) - f(^x)     f(x + ^x) - f(x - ^x)
    # f'(x) = ---- = lim ------------------- = -----------------------
    #          dx               ^x                      2^x
    #---------------------------------------------------------------------
    # 1-1. 손실함수에 대한 편미분공식 구현
    # 1-2. 손실함수를 1-1.공식으로 처리하면 편미분된다.
    # def derivative(f1, var):
    #   temp_var[index] = target_var + delta
    #   func_val_plus_delta = cost_func(temp_var)
    #   temp_var[index] = target_var - delta
    #   func_val_minus_delta = cost_func(temp_var)
    #   diff_val[row, column] = \
    #         (func_val_plus_delta - func_val_minus_delta) / (2 * delta)
    #---------------------------------------------------------------------
    # 1-2. 편미분대상 함수 : 손실함수 계산(트레이닝 데이터와 가설 데이터간의 차이의 평균)
    # def cost_func(x_data, t_data):
    #     y = np.dot(x_data, W) + b
    #     return (np.sum((t_data - y) ** 2)) / (len(x_data))
    ######################################################################

    # 최소화: 미분함수를 사용한 경사하강법: W -= learning_rate * 미분함수(변화량)
    #print("before WWWWW =",W )
    W -= learning_rate * derivative(f1, W)
    #print("after WWWWW =",W )

    #print("before bbbbbb =",b )
    b -= learning_rate * derivative(f1, b)
    #print("after bbbbbb =",b )

    if (step % 1001 == 0):
        print("step = ",step, "cost_value =", cost_func(x_data, t_data), "W =", W,"b =", b)

predict(np.array([430]))
