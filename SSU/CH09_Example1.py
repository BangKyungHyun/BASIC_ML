import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))
# 오차역전파 사용으로 , 수치미분 함수 정의 불필요
#from datetime import datetime  # datetime.now() 를 이용하여 학습 경과 시간 측정
import datetime

# Neural Network 클래스
class NeuralNetwork:
    ######################################################################
    # 신경망 초기화
    ######################################################################
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        # 각층의 노드와 학습률 learning rate 초기화
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 입력층 선형회귀 값 Z1, 출력값 A1 정의 (모두 행렬로 표시 - 오차역전파를 하기 위한 중간 결과)
        # 1차원 배열을 0으로 초기화하라는 의미
        # 예제 : np.zeros((2,2)) 는 2X2 2차원 배열을 0 으로 초기화 의미
        # [[0,0]
        #  [0,0]]
        # self.Z1 = np.zeros([1, input_nodes])  # 입력층은 선형회귀값과 출력값이 동일하여 Z1 정의 생략
        self.A1 = np.zeros([1, input_nodes])
        #print("self.Z1.shape = ",self.Z1.shape)
        print("self.A1.shape = ",self.A1.shape)

        # 은닉층 선형회귀 값 Z2, 출력값 A2 정의 (모두 행렬로 표시 - 오차역전파를 하기 위한 중간 결과)
        self.Z2 = np.zeros([1, hidden_nodes])
        self.A2 = np.zeros([1, hidden_nodes])   # sigmoid 처리 결과

        # 출력층 선형회귀 값 Z3, 출력값 A3 정의 (모두 행렬로 표시 - 오차역전파를 하기 위한 중간 결과)
        self.Z3 = np.zeros([1, output_nodes])
        self.A3 = np.zeros([1, output_nodes])    # sigmoid 처리 결과

        # 은닉층 가중치  W2 = (784 X 100) Xavier/He 방법으로 self.W2 가중치 초기화
        # randn은 기댓값이 0이고, 표준편차가 1인 가우시안 표준 정규 분포를 따르는 난수 생성
        # np.random.randn(m,n) : m x n 형태의 배열로 구성됨
        # W2 : input_nodes가 784, hidden_nodes 100으로 (784, 100) 배열 형태
        self.W2 = np.random.randn(self.input_nodes,self.hidden_nodes) \
                        / np.sqrt(self.input_nodes / 2)
        self.b2 = np.random.rand(self.hidden_nodes)

        # 출력층 가중치는 W3 = (100X10)  Xavier/He 방법으로 self.W3 가중치 초기화
        # W3 : hidden_nodes가 100, output_nodes 10으로 (100, 10) 배열 형태
        self.W3 = np.random.randn(self.hidden_nodes,self.output_nodes) \
                        / np.sqrt(self.hidden_nodes / 2)
        self.b3 = np.random.rand(self.output_nodes)
        print("self.Z2.shape = ",self.Z2.shape)
        print("self.A2.shape = ",self.A2.shape)
        print("self.Z3.shape = ",self.Z3.shape)
        print("self.A3.shape = ",self.A3.shape)
        print("self.W2.shape = ",self.W2.shape)
        print("self.b2.shape = ",self.b2.shape)
        print("self.W3.shape = ",self.W3.shape)
        print("self.b3.shape = ",self.b3.shape)
        '''
        self.A1.shape =  (1, 784)
        self.Z2.shape =  (1, 100)
        self.A2.shape =  (1, 100)
        self.Z3.shape =  (1, 10)
        self.A3.shape =  (1, 10)
        self.W2.shape =  (784, 100)
        self.b2.shape =  (100,)
        self.W3.shape =  (100, 10)
        self.b3.shape =  (10,)
        '''
    ######################################################################
    # Feed Forword 수행 (손실함수 계산까지 처리)
    ######################################################################
    def feed_forward(self):

        if step % 10000 == 0:
            print("<==== 2. feed_forward Start i = ", i, "step = ", step)

        delta = 1e-7  # log 무한대 발산 방지

        # 입력층 선형회귀 값 Z1 (1, 784) , 출력값 A1 (1, 784) 계산
        #self.Z1 = self.input_data  # self.Z1 은 실제 사용되지 않음
        self.A1 = self.input_data

        # 은닉층 선형회귀 값 Z2 (1, 100), 출력값 A2 (1, 100) 계산
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # sigmoid : 0~1 사이 값으로 변환
        self.A2 = sigmoid(self.Z2)

        # 출력층 선형회귀 값 Z3 (1, 10), 출력값 A3 (1, 10) 계산
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        y = self.A3 = sigmoid(self.Z3)

         #if step % 10000 == 0:
         #   print("self.Z1 =", self.Z1, "\nself.A1 =", self.A1)
         #   print("\nself.A1 =", self.A1, "\nself.W2 =", self.W2,
         #         "\nself.b2 =", self.b2, "\nself.Z2 =", self.Z2 )
         #   print("\nself.A2 =", self.A2, "\nself.W3 =", self.W3,
         #         "\nself.b3 =", self.b3, "\nself.Z3 =", self.Z3,
         #         "\nself.A3 =", self.A3)

        if step % 10000 == 0:
            print("<==== 2. feed_forward End i = ", i, "step = ", step)

        # cross entropy cost function
        return -np.sum(self.target_data * np.log(y + delta) +
                (1 - self.target_data) * np.log((1 - y) + delta))

    ######################################################################
    # 손실함수 값 계산(외부출력용)
    ######################################################################
    def loss_val(self):

        #print("<==== loss val Start ====>")

        delta = 1e-7  # log 무한대 발산 방지

        # 입력층 선형회귀 값 Z1, 출력값 A1 계산
        self.Z1 = self.input_data
        self.A1 = self.input_data

        # 은닉층 선형회귀 값 Z2, 출력값 A2 계산
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)

        # 출력층 선형회귀 값 Z3, 출력값 A3 계산
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        y = self.A3 = sigmoid(self.Z3)

        #print("<==== loss val End ====>")

        # cross entropy cost function
        return -np.sum(self.target_data * np.log(y + delta) +
                      (1 - self.target_data) * np.log((1 - y) + delta))

    ######################################################################
    #
    # 오차역전파를 이용하여 학습하는 함수
    #
    # 오차역전파 공식을 이용하여 가중치 W2, W3 바이어스 b2, b3값을 업데이트
    #
    # 입력 파라미터로 들어오는 입력 데이터(input data)와 정답 데이터(target data)를
    # 이용하여 feed forward를 수행하고
    # 출력층에서의 오차역전파 공식 W3 = W3 - a(학습률) * (A2.T*loss_3),
    # b3 = b3- a(학습률) * loss_3 수식을 적용하기 위해 가상의 손실 loss_3을 계산한
    # 다음 출력층 가중치 W3, 바이어스 b3을 업데이트 한다.
    #
    # 출력층과 마찬가지로 은닉층의 오차역전파 공식 W2 = W2 - a(학습률) * (A1.T*loss_2),
    # b2 = b2- a(학습률) * loss_2 수식을 적용하기 위해 가상의 손실 loss_2을 계산한
    # 다음 출력층 가중치 W2, 바이어스 b2을 업데이트 한다.
    ######################################################################
    def train(self, input_data, target_data):

        if step % 10000 == 0:
            print("<==== 1. Train Start i = ", i, "step = ", step)

        # input_data : 784 개, target_data : 10개
        self.input_data = input_data
        self.target_data = target_data

        # 먼저 feed forward 를 통해서 최종 출력값과 이를 바탕으로 현재의 loss value 계산
        loss_val111111 = self.feed_forward()

        # 출력층 loss 인 loss_3 구함
        loss_3 = (self.A3 - self.target_data) * self.A3 * (1 - self.A3)
        if step % 10000 == 0:
            print("<==== 1. Train loss_3 = ", loss_3, "self.A3.shape = ", self.A3.shape, "self.target_data.shape =", self.target_data.shape)

        # 출력층 가중치 W3, 출력층 바이어스 b3 업데이트
        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)
        self.b3 = self.b3 - self.learning_rate * loss_3
        if step % 10000 == 0:
            print("<==== 1. Train W3 = ", self.W3, "self.A2.T = ", self.A2.T)

        # 은닉층 loss 인 loss_2 구함
        loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1 - self.A2)

        # 은닉층 가중치 W2, 은닉층 바이어스 b2 업데이트
        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)
        self.b2 = self.b2 - self.learning_rate * loss_2

        if step % 10000 == 0:
            print("<==== 1. Train End i = ", i, "step = ", step)

    ######################################################################
    # 정확도 측정 method
    ######################################################################
    def accuracy(self, test_input_data, test_target_data):

        matched_list = []
        not_matched_list = []

        for index in range(len(test_input_data)):

            label = int(test_target_data[index])

            # one-hot encoding을 위한 데이터 정규화 (data normalize)
            data = (test_input_data[index] / 255.0 * 0.99) + 0.01

            # predict 를 위해서 vector 을 matrix 로 변환하여 인수로 넘겨줌
            predicted_num = self.predict(np.array(data, ndmin=2))

            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)

        print("Current Accuracy =", (len(matched_list) / (len(test_input_data))))

        return matched_list, not_matched_list

    ######################################################################
    # 미래 값 예측 method
    # input_data 는 행렬로 입력됨 즉, (1, 784) shape 을 가짐
    ######################################################################
    def predict(self, input_data):

        Z2 = np.dot(input_data, self.W2) + self.b2
        A2 = sigmoid(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        y = A3 = sigmoid(Z3)

        # MNIST 경우는 One-Hot-Encoding을 적용하기 때문에
        # 0 또는 1이 아닌 argmax()를 통해 최대 인덱스를 넘겨주여야 함
        predicted_num = np.argmax(y)

        return predicted_num

######################################################################
# Neural Network 객체 생성 및 정확도 검증
######################################################################

# 0~9 숫자 이미지가 784개의 숫자 (28X28) 로 구성되어 있는 training data 읽어옴
print("<==== Main Start ===>")

training_data = np.loadtxt('./mnist_train.csv', delimiter=',', dtype=np.float32)

print("training_data.shape = ", training_data.shape)
print("training_data[0,0] = ", training_data[0,0], ", len(training_data[0]) = ",
      len(training_data[0]))

# hyper-parameter
i_nodes = 784   # input nodes 갯수
h1_nodes = 100  # hidden nodes 갯수
o_nodes = 10    # output nodes 갯수
lr = 0.3        # learning rate
epochs = 1      # 반복횟수

# NeuralNetwork 객체 생성
print("<==== obj = NeuralNetwork(i_nodes, h1_nodes, o_nodes, lr) ===>")
obj = NeuralNetwork(i_nodes, h1_nodes, o_nodes, lr)

start_time = datetime.datetime.now()

for i in range(epochs):

    for step in range(len(training_data)):

        # input_data, target_data normalize하여 0~1사이의 값으로 변환
        input_data = ((training_data[step, 1:] / 255.0) * 0.99) + 0.01

        # o_nodes(10 형태 list)에 0으로 치환 후 0.01를 더한 값을 target_data로 대입함
        target_data = np.zeros(o_nodes) + 0.01

        # 60000 레코드별 첫번째 컬럼의 숫자에 해당하는 값을 0.99로 변환함
        target_data[int(training_data[step, 0])] = 0.99

        # 입력 데이터와 정답 데이터를 행렬로 만들어서 train() 메서드를 호출
        # 오차역전파 공식은 기본적으로 행렬 연산을 하기 때문에 벡터로 나타나는 입력 데이터와
        # 정답 데이터를 행렬로 바꾸어서 입력해 주어야 하므로
        # np.array ndmin=2 => 함수를 사용하며 2차원의 행렬로 변환함
        # [예제]
        # arr = np.array([1, 2, 3, 4], ndmin=5)
        # print(arr)
        # print('배열 차원수 :', arr.ndim)
        # 결과값:
        # [[[[[1 2 3 4]]]]]
        # 배열 차원수 : 5

        obj.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2))

        if step % 10000 == 0:
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

            # print("input_data =",input_data, "training_data[step, 1:] = ", training_data[step, 1:])
            #print("target_data[int(training_data[step, 0])] =",
            #      target_data[int(training_data[step, 0])],
            #      "int(training_data[step, 0] = ", int(training_data[step, 0]))
            # print("np.array(input_data, ndmin=2) =", np.array(input_data, ndmin=2),
            #       "np.array(target_data, ndmin=2) = ", np.array(target_data, ndmin=2))
            print(nowDatetime, "epochs = ", i, ", step = ", step, ", current loss_val = ",
                  obj.loss_val())

end_time = datetime.datetime.now()
print("\nelapsed time = ", end_time - start_time)

print("<==== Main End ===>")

######################################################################
# 검증 코드
######################################################################
# 0~9 숫자 이미지가 784개의 숫자 (28X28) 로 구성되어 있는 test data 읽어옴

print("==== def measure accuracy start ===")

test_data = np.loadtxt('./mnist_test.csv', delimiter=',', dtype=np.float32)

test_input_data = test_data[ : , 1: ]
test_target_data = test_data[ : , 0 ]

print("test_data.shape = ", test_data.shape)
print("test_data[0,0] = ", test_data[0,0], ", len(test_data[0]) = ", len(test_data[0]))

# measure accuracy
(true_list, false_list) = obj.accuracy(test_input_data, test_target_data)

print("true_list =", true_list)
print("false_list =", false_list)

print("==== def measure accuracy end ===")
