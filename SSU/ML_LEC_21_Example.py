import numpy as np

# 수치미분 함수

def numerical_derivative(f, x):
    delta_x = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)  # f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)  # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad

# sigmoid 함수

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogicGate:

    def __init__(self, gate_name, xdata, tdata):

        self.name = gate_name

        # 입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4, 2)  # 4개의 입력데이터 x1, x2 에 대하여 batch 처리 행렬
        self.__tdata = tdata.reshape(4, 1)  # 4개의 입력데이터 x1, x2 에 대한 각각의 계산 값 행렬

        # 2층 hidden layer unit : 6개 가정,  가중치 W2, 바이어스 b2 초기화
        self.__W2 = np.random.rand(2, 6)  # weight, 2 X 6 matrix
        self.__b2 = np.random.rand(6)

        # 3층 output layer unit : 1 개 , 가중치 W3, 바이어스 b3 초기화
        self.__W3 = np.random.rand(6, 1)
        self.__b3 = np.random.rand(1)

        # 학습률 learning rate 초기화
        self.__learning_rate = 1e-2

        print(self.name + " object is created")

    def feed_forward(self):  # feed forward 를 통하여 손실함수(cross-entropy) 값 계산

        delta = 1e-7  # log 무한대 발산 방지

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2  # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)  # 은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3  # 출력층의 선형회귀 값
        y = a3 = sigmoid(z3)  # 출력층의 출력

        # cross-entropy
        return -np.sum(
            self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(
                (1 - y) + delta))

    def loss_val(self):  # 외부 출력을 위한 손실함수(cross-entropy) 값 계산

        delta = 1e-7  # log 무한대 발산 방지

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2  # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)  # 은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3  # 출력층의 선형회귀 값
        y = a3 = sigmoid(z3)  # 출력층의 출력

        # cross-entropy
        return -np.sum(
            self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(
                (1 - y) + delta))

    # 수치미분을 이용하여 손실함수가 최소가 될때 까지 학습하는 함수
    def train(self):

        f = lambda x: self.feed_forward()

        print("Initial loss value = ", self.loss_val())

        for step in range(10001):

            self.__W2 -= self.__learning_rate * numerical_derivative(f,self.__W2)
            self.__b2 -= self.__learning_rate * numerical_derivative(f,self.__b2)
            self.__W3 -= self.__learning_rate * numerical_derivative(f,self.__W3)
            self.__b3 -= self.__learning_rate * numerical_derivative(f,self.__b3)

            if (step % 400 == 0):
                print("step = ", step, "  , loss value = ", self.loss_val())

    # query, 즉 미래 값 예측 함수
    def predict(self, xdata):

        z2 = np.dot(xdata, self.__W2) + self.__b2  # 은닉층의 선형회귀 값
        a2 = sigmoid(z2)  # 은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3  # 출력층의 선형회귀 값
        y = a3 = sigmoid(z3)  # 출력층의 출력

        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False

        return y, result

# AND Gate 객체 생성 및 training

xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 0, 0, 1])

and_obj = LogicGate("AND", xdata, tdata)

and_obj.train()

# AND Gate prediction

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for data in test_data:
    print(and_obj.predict(data))

# OR Gate 객체 생성 및 training

xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 1])

or_obj = LogicGate("OR", xdata, tdata)

or_obj.train()

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for data in test_data:
    print(or_obj.predict(data))

# NAND Gate 객체 생성 및 training

xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([1, 1, 1, 0])

nand_obj = LogicGate("NAND", xdata, tdata)

nand_obj.train()

# NAND Gate prediction

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for data in test_data:
    print(nand_obj.predict(data))

# XOR Gate 객체 생성

xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 0])

xor_obj = LogicGate("XOR", xdata, tdata)

xor_obj.train()

test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

for data in test_data:
    print(xor_obj.predict(data))
