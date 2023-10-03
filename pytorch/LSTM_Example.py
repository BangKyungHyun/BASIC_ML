# 아즈마 유키나가, 최재원·장건희 옮김, 「핵심 딥러닝 입문 RNN, LSTM, GRU, VAE, GAN 구현」, 책만, 2020, p.166~194.
# https://github.com/kyun1016/deep_learning_python/blob/master/2021_05_22_LSTM/LSTM%20%EB%AC%B8%EC%9E%A5%20%EC%83%9D%EC%84%B1%20(cpu).ipynb

import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import datetime

# -- 각 설정값 --
n_time = 20  # 시점의 수
n_mid = 128  # 은닉층

eta = 0.01  # 학습률
clip_const = 0.02  # 노름의 최댓값을 구하는 상수
beta = 2  # 확률분포 폭(다음 시점에 올 문자를 예측할 때 사용)
epoch = 200
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
    text = f.read()
print("문자 수:", len(text))  # len()으로 문자열의 문자 수도 출력 가능

# -- 문자와 인덱스 연결 --
chars_list = sorted(list(set(text)))  # set으로 문자 중복 제거
n_chars = len(chars_list)
print("문자 수 (중복없음) :", n_chars)

char_to_index = {}  # 문자가 키이고 인덱스가 값인 딕셔너리
index_to_char = {}  # 인덱스가 키이고 문자가 값인 딕셔너리

for i, char in enumerate(chars_list):
    char_to_index[char] = i
    index_to_char[i] = char

# -- 시계열로 나열된 문자와 다음 차례 문자 --
seq_chars = []
next_chars = []
for i in range(0, len(text) - n_time):
    seq_chars.append(text[i:i + n_time])
    next_chars.append(text[i + n_time])

# -- 입력과 정답을 원핫 인코딩으로 표시 --
input_data = np.zeros((len(seq_chars), n_time, n_chars), dtype=np.bool_)
correct_data = np.zeros((len(seq_chars), n_chars), dtype=np.bool_)
print('input_data length = ',len(input_data))
print('correct_data length = ',len(correct_data))

for i, chars in enumerate(seq_chars):
    # 정답을 원핫 인코딩으로 표시
    correct_data[i, char_to_index[next_chars[i]]] = 1
    for j, char in enumerate(chars):
        # 입력을 원핫 인코딩으로 표시
        input_data[i, j, char_to_index[char]] = 1

# -- LSTM층 --
class LSTMLayer:
    def __init__(self, n_upper, n):
        # 각 파라미터의 초깃값
        self.w = np.random.randn(4, n_upper, n) / np.sqrt(n_upper)
        self.v = np.random.randn(4, n, n) / np.sqrt(n)
        self.b = np.zeros((4, n))

    # y_prev, c_prev: 이전 시점의 출력과 기억 셀
    def forward(self, x, y_prev, c_prev):
        u = np.matmul(x, self.w) + np.matmul(y_prev, self.v) + self.b.reshape(4,1,-1)

        a0 = sigmoid(u[0])  # 망각 게이트
        a1 = sigmoid(u[1])  # 입력 게이트
        a2 = np.tanh(u[2])  # 새로운 기억
        a3 = sigmoid(u[3])  # 출력 게이트

        self.gates = np.stack((a0, a1, a2, a3))

        self.c = a0 * c_prev + a1 * a2  # 기억 셀
        self.y = a3 * np.tanh(self.c)  # 출력

    def backward(self, x, y, c, y_prev, c_prev, gates, grad_y, grad_c):
        a0, a1, a2, a3 = gates
        tanh_c = np.tanh(c)
        r = grad_c + (grad_y * a3) * (1 - tanh_c ** 2)

        # 각 delta
        delta_a0 = r * c_prev * a0 * (1 - a0)
        delta_a1 = r * a2 * a1 * (1 - a1)
        delta_a2 = r * a1 * (1 - a2 ** 2)
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

    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)

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
        # 자비에르 초기화 기반의 초깃값
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u) / np.sum(np.exp(u), axis=1).reshape(-1,
                                                               1)  # 소프트맥스 함수

    def backward(self, t):
        delta = self.y - t

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b


# -- 각 층의 초기화 --
lstm_layer = LSTMLayer(n_chars, n_mid)
output_layer = OutputLayer(n_mid, n_chars)


# -- 훈련 --
def train(x_mb, t_mb):
    # 순전파 LSTM층
    y_rnn = np.zeros((len(x_mb), n_time + 1, n_mid))
    c_rnn = np.zeros((len(x_mb), n_time + 1, n_mid))
    gates_rnn = np.zeros((4, len(x_mb), n_time, n_mid))
    y_prev = y_rnn[:, 0, :]
    c_prev = c_rnn[:, 0, :]

    for i in range(n_time):
        x = x_mb[:, i, :]
        lstm_layer.forward(x, y_prev, c_prev)

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
    for i in reversed(range(n_time)):
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
    y_prev = np.zeros((len(x_mb), n_mid))
    c_prev = np.zeros((len(x_mb), n_mid))

    for i in range(n_time):
        x = x_mb[:, i, :]
        lstm_layer.forward(x, y_prev, c_prev)
        y = lstm_layer.y
        y_prev = y
        c = lstm_layer.c
        c_prev = c

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
    prev_text = text[0:n_time]  # 입력
    created_text = prev_text  # 생성되는 텍스트
    print("Seed:", created_text)

    for i in range(200):  # 200자 문자 생성
        # 입력을 원핫 인코딩으로 표시
        x = np.zeros((1, n_time, n_chars))

        for j, char in enumerate(prev_text):
            x[0, j, char_to_index[char]] = 1

        # 다음 문자 예측
        y = predict(x)
        p = y[0] ** beta  # 확률분포 조정
        p = p / np.sum(p)  # p의 합을 1로
        next_index = np.random.choice(len(p), size=1, p=p)
        next_char = index_to_char[int(next_index[0])]
        created_text += next_char
        prev_text = prev_text[1:] + next_char

    print(created_text)
    print()  # 개행


error_record = []

# batch_size = 128
n_batch = len(input_data) // batch_size  # 1 에포크당 배치 개수

for i in range(epoch):
    # -- 학습 --
    index_random = np.arange(len(input_data))
    np.random.shuffle(index_random)  # 인덱스 임의 섞기
    for j in range(n_batch):
        # 미니 배치 구성
        mb_index = index_random[j * batch_size: (j + 1) * batch_size]
        x_mb = input_data[mb_index, :]
        t_mb = correct_data[mb_index, :]
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