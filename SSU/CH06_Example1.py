import numpy as np
from datetime import datetime

# sigmoid 함수
def sigmoid(x):
    return 1 / (1+np.exp(-x))

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

# LogicGate Class
class LogicGate:

    def __init__(self, gate_name, xdata, tdata):

        self.name = gate_name

        # 입력 데이터, 정답 데이터 초기화
        self.xdata = xdata.reshape(4, 2)
        self.tdata = tdata.reshape(4, 1)

        # 가중치 W, 바이어스 b 초기화
        self.W = np.random.rand(2, 1)
        self.b = np.random.rand(1)

        # 학습률 learning rate 초기화
        self.learning_rate = 1e-2

    # 손실함수
    def loss_func(self):

        delta = 1e-7  # log 무한대 발산 방지

        z = np.dot(self.xdata, self.W) + self.b
        y = sigmoid(z)

        # cross-entropy
        return -np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

    # 수치미분을 이용하여 손실함수가 최소가 될때 까지 학습하는 함수
    def train(self):

        f = lambda x: self.loss_func()
        print("Initial loss value = ", self.loss_func())

        for step in range(8001):
            self.W -= self.learning_rate * derivative(f, self.W)
            self.b -= self.learning_rate * derivative(f, self.b)

            if (step % 1000 == 0):
                print("step = ", step, "loss value = ", self.loss_func())

    # 미래 값 예측 함수
    def predict(self, input_data):

        z = np.dot(input_data, self.W) + self.b
        y = sigmoid(z)

        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False

        return y, result

print("AND Gate Start")
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 0, 0, 1])

AND_obj = LogicGate("AND_GATE", xdata, tdata)
AND_obj.train()

# AND Gate prediction
print(AND_obj.name, "\n")

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print("sigmoid_val =", sigmoid_val, "input_data =,",input_data, " = ", "logical_val = ",logical_val)

print("AND Gate end")

print("OR Gate Start")
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 1])

OR_obj = LogicGate("OR_GATE", xdata, tdata)
OR_obj.train()

# OR Gate prediction
print(OR_obj.name, "\n")

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data)
    print("sigmoid_val =", sigmoid_val, "input_data =,",input_data, " = ", "logical_val = ",logical_val)

print("OR Gate end")

print("NAND Gate Start")

xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([1, 1, 1, 0])

NAND_obj = LogicGate("NAND_GATE", xdata, tdata)
NAND_obj.train()

# NAND Gate prediction
print(NAND_obj.name, "\n")

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for input_data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(input_data)
    print("sigmoid_val =", sigmoid_val, "input_data =,",input_data, " = ", "logical_val = ",logical_val)

print("NAND Gate end")

print("XOR Gate Start")

xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 0])

XOR_obj = LogicGate("XOR_GATE", xdata, tdata)
# XOR Gate 를 보면, 손실함수 값이 2.7 근처에서 더 이상 감소하지 않는것을 볼수 있음
XOR_obj.train()

# XOR Gate prediction => 예측이 되지 않음
print(XOR_obj.name, "\n")

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for input_data in test_data:
    (sigmoid_val, logical_val) = XOR_obj.predict(input_data)
    print("sigmoid_val =", sigmoid_val, "input_data =,",input_data, " = ", "logical_val = ",logical_val)

print("XOR Gate end")

print("XOR 을 NAND + OR => AND 조합으로 계산 Start")
#######################################################
# XOR 을 NAND + OR => AND 조합으로 계산함
#######################################################
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

s1 = []  # NAND 출력
s2 = []  # OR 출력

new_input_data = []  # AND 입력
final_output = []    # AND 출력

for index in range(len(input_data)):
    s1 = NAND_obj.predict(input_data[index])  # NAND 출력
    s2 = OR_obj.predict(input_data[index])    # OR 출력

    new_input_data.append(s1[-1])  # AND 입력
    new_input_data.append(s2[-1])  # AND 입력

    (sigmoid_val, logical_val) = AND_obj.predict(np.array(new_input_data))

    final_output.append(logical_val)  # AND 출력, 즉 XOR 출력
    new_input_data = []  # AND 입력 초기화

for index in range(len(input_data)):
    print("input_data =,",input_data[index], " = ", "final_output = ",final_output[index])

print("XOR 을 NAND + OR => AND 조합으로 계산 end")
