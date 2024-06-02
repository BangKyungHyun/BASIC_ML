import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import datetime

#%matplotlib inline

def derivative(f, var):

    if var.ndim == 1:  # vector
        temp_var = var


        delta = 1e-5
        diff_val = np.zeros(var.shape)

        for index in range(len(var)):
            target_var = float(temp_var[index])
            temp_var[index] = target_var + delta
            func_val_plust_delta = f(temp_var)  # x+delta 에 대한 함수 값 계산
            temp_var[index] = target_var - delta
            func_val_minus_delta = f(temp_var)  # x-delta 에 대한 함수 값 계산
            diff_val[index] = (func_val_plust_delta - func_val_minus_delta) / (2 * delta)
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
                func_val_plus_delta = f(temp_var)  # x+delta 에 대한 함수 값 계산
                temp_var[row, column] = target_var - delta
                func_val_minus_delta = f(temp_var)  # x-delta 에 대한 함수 값 계산
                diff_val[row, column] = (func_val_plus_delta - func_val_minus_delta) / (2 * delta)
                temp_var[row, column] = target_var

        return diff_val

# sigmoid 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#MNIST_TEST CLASS

class MNIST_Test:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        #은닉층 가중치 W2 Xavier/He 방법으로 self.W2 가중치 초기화
        #Xavier/He : 각 층으로 들어오는 입력층 노드의 갯수의 제곱근을 이용해서 가중치를 초기화
        self.W2 = np.random.randn(self.input_nodes, self.hidden_nodes) / \
                  np.sqrt(self.input_nodes / 2)
        print("self.W2.shape = ", self.W2.shape)
        print("np.sqrt(self.input_nodes / 2) = ", np.sqrt(self.input_nodes / 2))
        self.b2 = np.random.rand(self.hidden_nodes)
        print("self.b2.shape = ", self.b2.shape)

        # 출력층 가중치는 Xavier/He 방법으로 self.W3 가중치 초기화
        self.W3 = np.random.randn(self.hidden_nodes, self.output_nodes) / \
                  np.sqrt(self.hidden_nodes / 2)
        print("self.W3.shape = ", self.W3.shape)

        self.b3 = np.random.rand(self.output_nodes)
        print("self.b3.shape = ", self.b3.shape)

        #학습률 Learning Rate 초기화
        self.learning_rate = learning_rate

    # feed forward 함수
    def feed_forward(self):
        delta = 1e-7       # log 무한대 발산 방지
        z2 = np.dot(self.input_data, self.W2) + self.b2

        #은닉층 선형 회귀 값
        a2 = sigmoid(z2)                   # 은닉층 출력
        z3 = np.dot(a2,self.W3) + self.b3  # 출력층 선형회귀값

        y = a3 = sigmoid(z3)
        return -np.sum(self.target_data * np.log(y+delta) +
                       (1-self.target_data) * np.log((1-y)+delta))

    # 수치 미분을 이용하여 손실함수가 최소가 될때까지 학습하는 함수
    def train(self,input_data,target_data):
        self.input_data = input_data
        self.target_data = target_data

        f = lambda x : self.feed_forward()

        self.W2 -= self.learning_rate * derivative(f, self.W2)
        self.b2 -= self.learning_rate * derivative(f, self.b2)
        self.W3 -= self.learning_rate * derivative(f, self.W3)
        self.b3 -= self.learning_rate * derivative(f, self.b3)

    # 미래 값 예측 함수
    def predict(self, input_data):
        z2 = np.dot(input_data, self.W2) + self.b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = a3 = sigmoid(z3)

        #MNIST 경우는 One-Hot Encoding을 적용하기 때문에
        #0 또는 1이 아닌 argmax()를 통해 최대 인덱스를 넘겨 주어야 함
        predicted_num = np.argmax(y)

        return predicted_num

    # 정확도 측정 함수
    def accuracy(self, input_data, target_data):
        matched_list = []
        not_matched_list = []

        for index in range(len(input_data)):
            label = int(target_data[index])

            #정규화
            data = (input_data[index, :] / 255.0 * 0.99) + 0.01
            predicted_num = self.predict(data)

            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
        print("Current Accuracy =", len(matched_list)/ (len(input_data)))
        return matched_list, unmatched_list

# MNIST_Test 객체 생성 및 정확도 검증
training_data = np.loadtxt('./mnist_train.csv', delimiter=',', dtype=np.float32)

i_nodes = training_data.shape[1]-1 # input nodes 갯수
h1_nodes = 30                      # hidden nodes 갯수
o_nodes = 10                       # output nodes 갯수
lr = 1e-2                          # learning rate
epochs = 1                         # 반복 횟수

#MNIST_Test 객체 생성
obj = MNIST_Test(i_nodes, h1_nodes, o_nodes, lr)

for step in range(epochs):
    for index in range(len(training_data)):
        # input_data, target_data normalize
        input_data = ((training_data[index, 1:] / 255.0) * 0.99) + 0.01
        target_data = np.zeros(o_nodes) + 0.01
        target_data[int(training_data[index, 0])] = 0.99

        obj.train(input_data, target_data)

        if (index % 1000 == 0):
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
            print(nowDatetime,"epochs =", step, "index =",index, "loss_value = ", obj.feed_forward())

test_data = np.loadtxt('./mnist_test.csv', delimiter=',', dtype=np.float32)

test_input_data = test_data[ :, 1:]
test_target_data = test_data[ :, 0]
(true_list_1, false_list1) = obj.accuracy(test_input_data, test_target_data)
