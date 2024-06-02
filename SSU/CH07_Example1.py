import numpy as np
from datetime import datetime

# sigmoid 함수
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def derivative(f, var):
    if var.ndim == 1:  # 1) b는 vector 이므로 1차원
        temp_var = var  #2) 원본 값 저장
        delta = 1e-5
        # 3) 미분계수 보관 변수 초기화
        diff_val = np.zeros(var.shape   )

        # 4) 벡터의 모든 열(column) 순서대로 반복함
        for index in range(len(var)):
            # 원본값에서 index 순서대로 할당
            target_var = float(temp_var[index])
            temp_var[index] = target_var + delta
            func_val_plus_delta = f(temp_var)
            temp_var[index] = target_var - delta
            func_val_minus_delta = f(temp_var)
            diff_val[index] = \
                (func_val_plus_delta - func_val_minus_delta) / (2 * delta)
            temp_var[index] = target_var
        return diff_val

    elif var.ndim == 2:  # matrix
        temp_var = var
        delta = 1e-5
        diff_val = np.zeros(var.shape)
        rows = var.shape[0]
        columns = var.shape[1]

        for row in range(rows):
            for column in range(columns):
                target_var = float(temp_var[row, column])
                temp_var[row, column] = target_var + delta
                func_val_plus_delta = f(temp_var)
                temp_var[row, column] = target_var - delta
                func_val_minus_delta = f(temp_var)
                diff_val[row, column] = \
                    (func_val_plus_delta - func_val_minus_delta) / (2 * delta)
                temp_var[row, column] = target_var

        return diff_val

class LogicGate:

    def __init__(self, gate_name, xdata, tdata):

        self.name = gate_name

        # 입력 데이터, 정답 데이터 초기화
        self.xdata = xdata.reshape(4, 2)  # 4개의 입력데이터 x1, x2 에 대하여 batch 처리 행렬
        self.tdata = tdata.reshape(4, 1)  # 4개의 입력데이터 x1, x2 에 대한 각각의 계산 값 행렬

        # 2층 hidden layer unit : 6개 가정,  가중치 W2, 바이어스 b2 초기화
        self.W2 = np.random.rand(2, 6)  # weight, 2 X 6 matrix
        self.b2 = np.random.rand(6)
        print("self.W2 =", self.W2)
        print("self.b2 =", self.b2)

        # 3층 output layer unit : 1 개 , 가중치 W3, 바이어스 b3 초기화
        self.W3 = np.random.rand(6, 1)
        self.b3 = np.random.rand(1)
        print("self.W3 =", self.W3)
        print("self.b3 =", self.b3)

        # 학습률 learning rate 초기화
        self.learning_rate = 1e-2

        print(self.name + " object is created")

    def feed_forward(self):  # feed forward 를 통하여 손실함수(cross-entropy) 값 계산

        delta = 1e-7  # log 무한대 발산 방지

        z2 = np.dot(self.xdata, self.W2) + self.b2  # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)  # 은닉층의 출력

        z3 = np.dot(a2, self.W3) + self.b3  # 출력층의 선형회귀 값

        y = a3 = sigmoid(z3)  # 출력층의 출력

        # cross-entropy
        return -np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))


    def loss_val(self):  # 외부 출력을 위한 손실함수(cross-entropy) 값 계산

        delta = 1e-7  # log 무한대 발산 방지

        z2 = np.dot(self.xdata, self.W2) + self.b2  # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)                            # 은닉층의 출력

        z3 = np.dot(a2, self.W3) + self.b3          # 출력층의 선형회귀 값
        y = a3 = sigmoid(z3)                        # 출력층의 출력

        # cross-entropy
        return -np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

        # 수치미분을 이용하여 손실함수가 최소가 될때 까지 학습하는 함수

    def train(self):

        f = lambda x: self.feed_forward()

        print("Initial loss value = ", self.loss_val())

        for step in range(200001):

            self.W2 -= self.learning_rate * derivative(f, self.W2)
            self.b2 -= self.learning_rate * derivative(f, self.b2)
            self.W3 -= self.learning_rate * derivative(f, self.W3)
            self.b3 -= self.learning_rate * derivative(f, self.b3)

            if (step % 1000 == 0):
                print("step = ", step, "  , loss value = ", self.loss_val())

    # query, 즉 미래 값 예측 함수
    def predict(self, input_data):

        z2 = np.dot(input_data, self.W2) + self.b2  # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)                            # 은닉층의 출력

        z3 = np.dot(a2, self.W3) + self.b3          # 출력층의 선형회귀 값
        y = a3 = sigmoid(z3)                        # 출력층의 출력

        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False

        return y, result

# XOR Gate 객체 생성
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 0])

print("===== LogicGate(XOR, xdata, tdata) Start =====")
xor_obj = LogicGate("XOR", xdata, tdata)
print("===== LogicGate(XOR, xdata, tdata) End =====")

print("===== xor_obj.train() Start =====")
xor_obj.train()
print("===== xor_obj.train() End =====")

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for data in test_data:
    (sigmoid_val, logical_val) = xor_obj.predict(data)
    print(data, " = ", logical_val)
