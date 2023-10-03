# 아즈마 유키나가, 최재원·장건희 옮김, 「핵심 딥러닝 입문 RNN, LSTM, GRU, VAE, GAN 구현」, 책만, 2020, p.166~194.
# https://github.com/kyun1016/deep_learning_python/blob/master/2021_05_22_LSTM/LSTM%20%EB%AC%B8%EC%9E%A5%20%EC%83%9D%EC%84%B1%20(cpu).ipynb

import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import datetime

# -- 각 설정값 --
input_char_cnt = 20  # 시점의 수(입력으로 사용되는 문자의 수)
hidden_layer_cnt = 128  # 은닉층

# eta = 0.01  # 학습률
eta = 0.01  # 학습률
clip_const = 0.02  # 노름의 최댓값을 구하는 상수
beta = 2  # 확률분포 폭(다음 시점에 올 문자를 예측할 때 사용)
epoch = 1
batch_size = 128

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clip_grad(grads, max_norm):

    norm = np.sqrt(np.sum(grads * grads))
    r = max_norm / norm
    if r < 1:
        clipped_grads = grads * r
    else:
        clipped_grads = grads
    return clipped_grads

# -- 훈련용 텍스트 --
with open("human_problem.txt", mode="r", encoding="utf-8-sig") as f:
# with open("human_problem_short.txt", mode="r", encoding="utf-8-sig") as f:
    text = f.read()
print("문자 수:", len(text))  # len()으로 문자열의 문자 수도 출력 가능
# 문자 수: 60322

# -- 문자와 인덱스 연결 --
chars_list = sorted(list(set(text)))  # set으로 문자 중복 제거
unique_char_cnt = len(chars_list)

print("중복 제거 문자 수 :", unique_char_cnt)
# 중복 제거 문자 수 : 980

char_to_index = {}  # 문자가 키이고 인덱스가 값인 딕셔너리
index_to_char = {}  # 인덱스가 키이고 문자가 값인 딕셔너리

for i, char in enumerate(chars_list):
    char_to_index[char] = i
    index_to_char[i] = char

print(char_to_index)
# {'\n': 0, ' ': 1, '!': 2, ',': 3, '-': 4, '.': 5, '?': 6, '―': 7, '‘': 8, '’': 9, '…': 10, '가': 11, '각': 12, '간': 13, '갈': 14, '감': 15, '갑': 16, '값': 17, '갔': 18, '강': 19, '갖': 20, '같': 21, '갚': 22, '갛': 23, '개': 24, '갸': 25, '걀': 26,
print(index_to_char)
# {0: '\n', 1: ' ', 2: '!', 3: ',', 4: '-', 5: '.', 6: '?', 7: '―', 8: '‘', 9: '’', 10: '…', 11: '가', 12: '각', 13: '간', 14: '갈', 15: '감', 16: '갑', 17: '값', 18: '갔', 19: '강', 20: '갖', 21: '같', 22: '갚', 23: '갛', 24: '개', 25: '갸', 26: '걀',

# -- 시계열로 나열된 문자와 다음 차례 문자 --
seq_chars = []
next_chars = []
# len(text) = 60322  - 전체 문자의 수
# input_char_cnt = 20  # 시점의 수(입력 문자의 수)
# 전체 문자의 수에서 입력 문자의 수를 빼 주면 시계열로 나열된 문자가 계산됨
for i in range(0, len(text) - input_char_cnt): # 0, 60302
    seq_chars.append(text[i:i + input_char_cnt])
    next_chars.append(text[i + input_char_cnt])

#print(seq_chars)
# print(next_chars)
# ['이 산등에 올라서면 용연 동네는 저렇', ' 산등에 올라서면 용연 동네는 저렇게', '산등에 올라서면 용연 동네는 저렇게 ', '등에 올라서면 용연 동네는 저렇게 뻔', '에 올라서면 용연 동네는 저렇게 뻔히', ' 올라서면 용연 동네는 저렇게 뻔히 ', '올라서면 용연 동네는 저렇게 뻔히 들',
# ['게', ' ', '뻔', '히', ' ', '들', '여', '다', '볼', ' ']

# print('len(seq_chars)  = ',len(seq_chars))
# len(seq_chars)  =  60302
# print('seq_chars[0]  = ',seq_chars[0])
# seq_chars[0]  =  이 산등에 올라서면 용연 동네는 저렇

print("unique_char_cnt =>문자 수 (중복없음) :", unique_char_cnt)
# unique_char_cnt =>문자 수 (중복없음) : 980
################################################################################
# 20개 입력 문자가 들어왔을 때 1개 정답 문자가 나오는 자료구조 구현
################################################################################

# -- 입력과 정답을 원핫 인코딩으로 표시 --
# 60302 * 20 * 980 - 샘플 수 * 시점 수(입력 문자 수) * 사용된 문자의 수
input_data = np.zeros((len(seq_chars), input_char_cnt, unique_char_cnt), dtype=np.bool_)
# print('input_data.shape = ', input_data.shape)
# print('input_data[0].shape = ', input_data[0].shape)
# print('input_data[0,0].shape = ', input_data[0, 0].shape)
# print('input_data[0,0,0].shape = ', input_data[0, 0, 0].shape)
# input_data.shape = (60302, 20, 980)
# input_data[0].shape = (20, 980)
# input_data[0, 0].shape = (980,)
# input_data[0, 0, 0].shape = ()

# input_data =  [[[False False False ... False False False]
#   [False False False ... False False False]
#   [False False False ... False False False]
#   ...

# 60302 * 980  샘플 수 * 사용된 문자의 수
correct_data = np.zeros((len(seq_chars), unique_char_cnt), dtype=np.bool_)
# print('correct_data = ',correct_data)
# correct_data =  [[False False False ... False False False]
#  [False False False ... False False False]
#  [False False False ... False False False]
#  ...
#  [False False False ... False False False]
#  [False False False ... False False False]
#  [False False False ... False False False]]

print('input_data shape = ',input_data.shape)
print('correct_data shape = ',correct_data.shape)
# input_data shape =  (60302, 20, 980)
# correct_data shape =  (60302, 980)

for i, chars in enumerate(seq_chars): # 60302개 문장

    # print (' i= ', i)
    # print (' chars= ', chars)
    # i=  0
    # chars=  이 산등에 올라서면 용연 동네는 저렇
    # i=  1
    # chars=   산등에 올라서면 용연 동네는 저렇게
    # i=  2
    # chars=  산등에 올라서면 용연 동네는 저렇게

    # 정답을 원핫 인코딩으로 표시
    correct_data[i, char_to_index[next_chars[i]]] = 1
    # print (' char_to_index[next_chars[i]]= ', char_to_index[next_chars[i]])

    # char_to_index[next_chars[i]]=  7
    # char_to_index[next_chars[i]]=  1
    # char_to_index[next_chars[i]]=  33

    for j, char in enumerate(chars):
        # 입력을 원핫 인코딩으로 표시
        input_data[i, j, char_to_index[char]] = 1   # 60302 *
        # print('input_data[0,0] = ',input_data[0,0])
        # -전체 문장(60322개 단어)을 표현하려면 20자 단위로 끊어서 표현 할려면 60302 개 배열 필요
        # [이 산등에 올라서면 용연 동네는 저렇]을 원핫벡터로 표현 할려면 각 글자(이, 산 등)별로 980개 1차원 벡터 필요하고
        # 문장의 길이가 20개 이므로 각 글자로 980개 1차원 벡터가 20개 필요하며 20*980으로
        # [이 산등에 올라서면 용연 동네는 저렇]를 표현함
        # ---------------------980 개---------------
        #      |T F F F F F F F F F F F F F F  이
        # 20개 |F T F F F F  F F F F F F F F
        #      |F F T F F F T F F F F F F F F  산
        #      |F F F T F F T F F F F F F F F  등
        #      |F F F F T F T F F F F F F F F  에

# -- LSTM층 --
class LSTMLayer:
    #  중복 제거된 문자 수(980), 은닉층 갯수(128) 
    def __init__(self, n_upper, n):
        # 각 파라미터의 초깃값

        # print('def __init__(self, n_upper, n) => n_upper = ', n_upper)
        # def __init__(self, n_upper, n) => n_upper =  980
        # print('def __init__(self, n_upper, n) => n ', n)
        # def __init__(self, n_upper, n) =>  n =  128

        self.w = np.random.randn(4, n_upper, n) / np.sqrt(n_upper)
        self.v = np.random.randn(4, n, n) / np.sqrt(n)
        self.b = np.zeros((4, n))

        print('self.w shape= ', self.w.shape)
        # self.w shape = (4, 980, 128)
        print('self.v shape= ', self.v.shape)
        # self.v shape = (4, 128, 128)
        print('self.b shape= ', self.b.shape)
        # self.b shape = (4, 128)

        # print('def __init__(self, n_upper, n) => self.w = ', self.w)
        # def __init__(self, n_upper, n) => self.w =
        # [[[ 0.00386036 -0.00209992 -0.04440902 ...  0.01993168  0.0192133
        #    -0.05684117]
        #   [ 0.02911864  0.01227388 -0.04737554 ...  0.04404453  0.017816
        #     0.05557986]
        #   [ 0.0109439  -0.01346285 -0.02497065 ...  0.01871527 -0.01475863
        #    -0.01097135]
        #   ...
        #   [ 0.06698153  0.04974968 -0.0319001  ... -0.00861584 -0.03473232
        #    -0.03246596]
        #   [ 0.03337388 -0.08460179 -0.0570803  ...  0.02426883 -0.02115659
        #     0.00513722]
        #   [ 0.015       0.01132891 -0.04316681 ...  0.02549255  0.01467493
        #    -0.00725941]]

        # print('def __init__(self, n_upper, n) => self.v = ', self.v)
        # def __init__(self, n_upper, n) => self.v =
        # [[[-0.03115449  0.0519018  -0.04153361 ...  0.04632568  0.22268866
        #     0.12813798]
        #   [ 0.07894218 -0.14404898 -0.01441008 ...  0.04107065 -0.142468
        #     0.08631654]
        #   [-0.02085287  0.04378927 -0.23237386 ... -0.1129165  -0.04677601
        #     0.05107761]
        #   ...
        #   [-0.07615161  0.02317473  0.06234304 ... -0.13442549 -0.00194572
        #     0.00039699]
        #   [ 0.0703535   0.07801443  0.19983976 ... -0.07738005  0.06207177
        #    -0.00501992]
        #   [ 0.05196663  0.03641155 -0.07917152 ...  0.09199845 -0.11893539
        #    -0.06763681]]

        # print('def __init__(self, n_upper, n) => self.b = ', self.b)
        # def __init__(self, n_upper, n) => self.b =
        # [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0.]

    def forward(self, x, y_prev, c_prev):
        # x : 입력, y_prev : 이전 시점의 출력, c_prev: 이전 시점의 기억 셀

        # 편향은 reshape을 통해 차원을 맞춘 뒤 브로드캐스팅을 이용해 계산, 이 계산으로 배치 내의 모든 샘플에서 같은 계산이 수행
        u = np.matmul(x, self.w) + np.matmul(y_prev, self.v) + self.b.reshape(4,1,-1)
        a0 = sigmoid(u[0])  # 망각 게이트
        a1 = sigmoid(u[1])  # 입력 게이트
        a2 = np.tanh(u[2])  # 새로운 기억
        a3 = sigmoid(u[3])  # 출력 게이트
        # print('def forward(self, x, y_prev, c_prev): => a0 = ', a0)
        # print('u shape = ', u.shape)
        # u shape = (4, 128, 128)
        # print('a0 shape = ', a0.shape)
        # a0 shape = (128, 128)

        self.gates = np.stack((a0, a1, a2, a3))
        # print('self.gates shape = ', self.gates.shape)
        # self.gates shape = (4, 128, 128)

        self.c = a0 * c_prev + a1 * a2 # 기억 셀 = 망각게이트*이전 시점의 기억셀 + 입력게이트*새로운 기억(candidate)
        self.y = a3 * np.tanh(self.c)  # 출력 = 출력게이트 * tanh(기억 셀)

    def backward(self, x, y, c, y_prev, c_prev, gates, grad_y, grad_c):

        # a0 : 망각 게이트, a1 : 입력 게이트, a2 : 새로운 기억, a3 : 출력 게이트
        # x : 입력, y_prev : 이전 시점의 출력, c: 기억 셀
        # grad_y : 출력 기울기, grad_c : 기억셀 기울기
        # w, v : 가중치 (4개의 행렬을 포함하는 배열)
        a0, a1, a2, a3 = gates
        tanh_c = np.tanh(c)   # 기억 셀
        # r = 기억 셀 기울기 +(출력 기울기 * 출력 게이트) * ( 1- tanh(기억셀) ** 2)
        r = grad_c + (grad_y * a3) * (1 - tanh_c ** 2)

        # 각 delta
        # delta_a0 = r * 이전 시점의 기억 셀 * 망각 게이트 * ( 1 - 망각 케이트)
        delta_a0 = r * c_prev * a0 * (1 - a0)

        # delta_a1 = r * 새로운 기억 * 입력 게이트 * ( 1 - 입력 케이트)
        delta_a1 = r * a2 * a1 * (1 - a1)

        # delta_a2 = r * 입력 게이트 * ( 1 - 새로운 기억 **2)
        delta_a2 = r * a1 * (1 - a2 ** 2)

        # delta_a3 = 출력 기울기 * tanh(기억셀) * 출력 게이트 * (1 - 출력 게이트)
        delta_a3 = grad_y * tanh_c * a3 * (1 - a3)

        deltas = np.stack((delta_a0, delta_a1, delta_a2, delta_a3))

        # 각 파라미터의 기울기
        self.grad_w += np.matmul(x.T, deltas)
        self.grad_v += np.matmul(y_prev.T, deltas)
        self.grad_b += np.sum(deltas, axis=1)

        # x 기울기
        grad_x = np.matmul(deltas, self.w.transpose(0, 2, 1))
        self.grad_x = np.sum(grad_x, axis=0)

        # y_prev 기울기
        grad_y_prev = np.matmul(deltas, self.v.transpose(0, 2, 1))
        self.grad_y_prev = np.sum(grad_y_prev, axis=0)

        # c_prev 기울기
        self.grad_c_prev = r * a0

    #  누적된 기울기를 0으로 리셋하는 메서드
    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)

    #  파라미터를 갱신하는 update 메서드
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.v -= eta * self.grad_v
        self.b -= eta * self.grad_b

    def clip_grads(self, clip_const):
        self.grad_w = clip_grad(self.grad_w,
                                clip_const * np.sqrt(self.grad_w.size))
        self.grad_v = clip_grad(self.grad_v,
                                clip_const * np.sqrt(self.grad_v.size))


# -- 전결합 출력층 --
class OutputLayer:

    def __init__(self, n_upper, n):
        # 샤비에르 초기화 기반의 초깃값
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1).reshape(-1,1)  # 소프트맥스 함수

    def backward(self, t):
        delta = self.y - t  

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

# -- 각 층의 초기화 --
#  중복 제거된 문자 수, 은닉층 갯수 
lstm_layer = LSTMLayer(unique_char_cnt, hidden_layer_cnt)  
output_layer = OutputLayer(hidden_layer_cnt, unique_char_cnt)

# -- 훈련 --
def train(x_mb, t_mb):

    # 순전파 LSTM층
    # print('len(x_mb) = ',len(x_mb))
    # print('input_char_cnt + 1 = ',input_char_cnt + 1)
    # print('hidden_layer_cnt = ',hidden_layer_cnt)
    # len(x_mb) = 128
    # input_char_cnt + 1 = 21
    # hidden_layer_cnt = 128
    y_rnn = np.zeros((len(x_mb), input_char_cnt + 1, hidden_layer_cnt))
    # print('y_rnn =\n ',y_rnn)
    # y_rnn =
    #   [[[0. 0. 0. ... 0. 0. 0.]
    #   [0. 0. 0. ... 0. 0. 0.]
    #   [0. 0. 0. ... 0. 0. 0.]
    #   ...
    #   [0. 0. 0. ... 0. 0. 0.]
    #   [0. 0. 0. ... 0. 0. 0.]
    #   [0. 0. 0. ... 0. 0. 0.]]
    c_rnn = np.zeros((len(x_mb), input_char_cnt + 1, hidden_layer_cnt))
    # print('c_rnn =\n ',c_rnn)
    # c_rnn =
    #   [[[0. 0. 0. ... 0. 0. 0.]
    #   [0. 0. 0. ... 0. 0. 0.]
    #   [0. 0. 0. ... 0. 0. 0.]
    #   ...
    #   [0. 0. 0. ... 0. 0. 0.]
    #   [0. 0. 0. ... 0. 0. 0.]
    #   [0. 0. 0. ... 0. 0. 0.]]
    gates_rnn = np.zeros((4, len(x_mb), input_char_cnt, hidden_layer_cnt))
    # print('gates_rnn = ',gates_rnn)
    # gates_rnn =
    # [[[[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]
    #
    #   [[0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    ...
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]
    #    [0. 0. 0. ... 0. 0. 0.]]

    y_prev = y_rnn[:, 0, :]
    c_prev = c_rnn[:, 0, :]
    # print('y_prev =\n ',y_prev)
    # print('c_prev =\n ',c_prev)

    # y_prev =
    #   [[0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]
    #  ...
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]]
    # c_prev =
    #   [[0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]
    #  ...
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]
    #  [0. 0. 0. ... 0. 0. 0.]]

    for i in range(input_char_cnt):   # 20
        x = x_mb[:, i, :]
        # print('x_mb[:, i, :] =\n ',x_mb[:, i, :])
        # x_mb[:, i, :] =
        #   [[ True False False ... False False False]
        #  [False  True False ... False False False]
        #  [ True False False ... False False False]
        #  ...
        #  [False False False ... False False False]
        #  [False False False ... False False False]
        #  [False  True False ... False False False]]

        lstm_layer.forward(x, y_prev, c_prev)

        # print('lstm_layer.forward(x, y_prev, c_prev) = ', lstm_layer.forward(x, y_prev, c_prev))

        y = lstm_layer.y
        y_rnn[:, i + 1, :] = y
        y_prev = y

        c = lstm_layer.c
        c_rnn[:, i + 1, :] = c
        c_prev = c

        gates = lstm_layer.gates
        gates_rnn[:, :, i, :] = gates

    # 순전파 출력층
    output_layer.forward(y)

    # 역전파 출력층
    output_layer.backward(t_mb)
    grad_y = output_layer.grad_x
    grad_c = np.zeros_like(lstm_layer.c)

    # 역전파 LSTM층
    lstm_layer.reset_sum_grad()

    for i in reversed(range(input_char_cnt)):
        x = x_mb[:, i, :]
        y = y_rnn[:, i + 1, :]
        c = c_rnn[:, i + 1, :]
        y_prev = y_rnn[:, i, :]
        c_prev = c_rnn[:, i, :]
        gates = gates_rnn[:, :, i, :]
        lstm_layer.backward(x, y, c, y_prev, c_prev, gates, grad_y, grad_c)
        grad_y = lstm_layer.grad_y_prev
        grad_c = lstm_layer.grad_c_prev

    # 파라미터 갱신
    lstm_layer.clip_grads(clip_const)
    lstm_layer.update(eta)
    output_layer.update(eta)

# -- 예측 --
def predict(x_mb):
    # 순전파 LSTM층
    y_prev = np.zeros((len(x_mb), hidden_layer_cnt))
    c_prev = np.zeros((len(x_mb), hidden_layer_cnt))

    for i in range(input_char_cnt):  # 20 개
        x = x_mb[:, i, :]
        #print('def predict = x \n', x)
        # [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
        lstm_layer.forward(x, y_prev, c_prev)
        # print('def predict = x, y_prev, c_prev, lstm_layer.forward(x, y_prev, c_prev) = \n', x, y_prev, c_prev, lstm_layer.forward(x, y_prev, c_prev))

        y = lstm_layer.y
        y_prev = y
        c = lstm_layer.c
        c_prev = c
        # print('def predict = y, y_prev, c, c_prev = \n', y, y_prev, c, c_prev)

    # 순전파 출력층
    output_layer.forward(y)
    return output_layer.y

# -- 오차 계산 --
def get_error(x, t):
    limit = 1000
    if len(x) > limit:  # 측정 샘플 수 최댓값 설정
        index_random = np.arange(len(x))
        np.random.shuffle(index_random)
        x = x[index_random[:limit], :]
        t = t[index_random[:limit], :]
    y = predict(x)
    # 교차 엔트로피 오차
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def create_text():
    #           text(전체 문자) : 63022 문자, input_char_cnt(입력문자) 20 문자
    prev_text = text[0:input_char_cnt]  # 입력 문자 20개
    created_text = prev_text            # 생성되는 텍스트
    print("Seed:", created_text)

    for i in range(200):  # 200자 출력 문자 생성
        # 입력을 원핫 인코딩으로 표시
        #  [1,20,980]
        x = np.zeros((1, input_char_cnt, unique_char_cnt))

        for j, char in enumerate(prev_text): # 20개 문자
            x[0, j, char_to_index[char]] = 1
            # print('create_text x, j, char_to_index[char], char = \n', x, j, char_to_index[char], char)
        # 다음 문자 예측
        # print('create_text  x = \n', x)
        y = predict(x)
        p = y[0] ** beta  # 확률분포 조정 beta = 2
        # np.sum() 함수 기본 사용법 : 기본 사용법은 array 내 전체 값들의 합을 구하는 방법
        p = p / np.sum(p)  # p의 합을 1로
        print('create_text y[0], p = \n', y[0], p)

        # numpy.random.choice(a, size=None, replace=True, p=None)
        # a : 1차원 배열 또는 정수 (정수인 경우, np.arange(a) 와 같은 배열 생성)
        # size : 정수 또는 튜플(튜플인 경우, 행렬로 리턴됨. (m, n, k) -> m * n * k), optional
        # replace : 중복 허용 여부, boolean, optional
        # p : 1차원 배열, 각 데이터가 선택될 확률, optional
        # numpy.random.choice(5, 3, True)
        # - 0 이상 5 미만인 정수 중 3개를 출력한다. (중복 허용)
        # numpy.random.choice(5, 3, False)
        # - 0 이상 5 미만인 정수 중 3개를 출력한다. (중복 비허용)

        next_index = np.random.choice(len(p), size=1, p=p)
        next_char = index_to_char[int(next_index[0])]
        created_text += next_char
        prev_text = prev_text[1:] + next_char

    print(created_text)
    print()  # 개행

error_record = []

# input_data length =  60302
# batch_size = 128
n_batch = len(input_data) // batch_size  # 1 에포크당 배치 개수
print('n_batch =', n_batch)
# n_batch = 471

for i in range(epoch): 
    # -- 학습 --
    # np.arange(시작점, 끝점, step size)
    # np.arange(10)
    # # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #
    # np.arange(1, 15, 2)
    # # array([ 1,  3,  5,  7,  9, 11, 13])
    #
    # np.arange(9, -2, -1.5)
    # # array([ 9. ,  7.5,  6. ,  4.5,  3. ,  1.5,  0. , -1.5])
    index_random = np.arange(len(input_data))
    print('np.arange(len(input_data)) =', np.arange(len(input_data)))
    print('index_random =', index_random)
    # np.arange(len(input_data)) = [0     1     2... 60299 60300 60301]
    np.random.shuffle(index_random)  # 인덱스 임의 섞기
    print('np.random.shuffle(index_random) =', np.random.shuffle(index_random))
    print('index_random =', index_random)
    # np.random.shuffle(index_random) = None
    # index_random = [30948 45238 24615... 48699 57547 12631]

    for j in range(n_batch):   #471개 배치
        # 미니 배치 구성
        mb_index = index_random[j * batch_size: (j + 1) * batch_size]
        #                       0 * 128:        1       * 128
        # print('mb_index =', mb_index)
        # mb_index = [24177 21664 36328 14991 38631 52332 24017  5022 22642 29345 32914  2649  총 128개
        #             1521 38486 57774 25629 40707 34061 46882 10720 50632  4357  3411  6335
        #             44708 24509 23412 48344 46545 10264  4334   300 44890 20735 16711 25218
        #             17563 17895 38398 54558 10544 59578  8503 36163 37909 10336 12159 19009
        #             22944  9193 26841  7572 26996 46512 57804 23738  8585  6188 45570 52655
        #             35835 22266 35862  7117  2929 29435 25083 10629 59814 24961 19262 43831
        #             30514  4970  8778 35244  8888  7140  8089 18763 49410 15521 6081 10527
        #             306 21621 36675 52100 58525   398 52361 40192 13953  9763   24278  5862
        #             16834 36062 27731 44589  2254  7254 29646 43865 12008 37981 50490 56354
        #             15608  5227 14198 34206 49587 21426 36079 59150 11544 34458 47376 20443
        #             10578 56242 24585 21182  8669  8951 33873  1536]

        x_mb = input_data[mb_index, :]
        # print('x_mb =', x_mb)
        # x_mb =
        # [[[False False False ... False False False]
        #   [False False False ... False False False]
        #   [False False False ... False False False]
        #   ...
        #   [False False False ... False False False]
        #   [ True False False ... False False False]
        #   [False False False ... False False False]]

        t_mb = correct_data[mb_index, :]
        # print('t_mb =', t_mb)
        # t_mb =
        # [[False False False ... False False False]
        #  [False False False ... False False False]
        #  [False False False ... False False False]
        #  ...
        #  [False  True False ... False False False]
        #  [False False False ... False False False]
        #  [False False False ... False False False]]

        train(x_mb, t_mb)

        # -- 경과 표시 --
        if (j + 1) % 100 == 0:
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
            print(nowDatetime,"Epoch: " + str(i + 1) + "/" + str(epoch) + " " + str(
            j + 1) + "/" + str(n_batch))

    # -- 오차 계산 --
    error = get_error(input_data, correct_data)
    error_record.append(error)
    print(" Loss: " + str(error))

    # -- 경과 표시 --
    create_text()

plt.plot(range(1, len(error_record) + 1), error_record, label="error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
plt.show()