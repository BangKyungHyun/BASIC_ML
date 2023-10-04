from typing import List, Any

from tqdm import tqdm  # 반복문에서 진행률을 보여주는 기능 수행
import numpy as np
import re

##### Data #####
data = "나라의 말이 중국과 달라 문자와 서로 통하지 아니하기에 이런 까닭으로 어리석은 백성이 이르고자 할 바가 있어도 마침내 제 뜻을 능히 펴지 못할 사람이 많으니라 내가 이를 위해 가엾이 여겨 새로 스물여덟 글자를 만드노니 사람마다 하여 쉬이 익혀 날로 씀에 편안케 하고자 할 따름이니라"

################################################################################
#데이터를 preprocessing 해주는 부분입니다
################################################################################
# re.sub(pattern, replace, text) : text 중 pattern에 해당하는 부분을 replace로 대체한다.
# data = re.sub('[^가-힣]', ' ', data): 자모만 아닌 한글만 남기기(공백제거)
############################################################################
# my_str = "안녕하세요 ㅎㅎ. Hello World! 12345?"
#
# kor_str = re.sub(r"[^ㄱ-ㅣ가-힣\s]", "", my_str) # 한글 + 공백만 남기기
# not_kor_str = re.sub(r"[ㄱ-ㅣ가-힣]", "", my_str) # 한글만 제거하기
# not_zamo_str = re.sub(r"[^가-힣]", "", my_str) # 자모가 아닌 한글만 남기기(공백 제거)
#
# print(kor_str) # 안녕하세요 ㅎㅎ
# print(not_kor_str) #  . hello world! 12345?
# print(not_zamo_str) # 안녕하세요
############################################################################

def data_preprocessing(data):
    data = re.sub('[^가-힣]', ' ', data)
    tokens = data.split()
    #  list(set())을 이용한 중복제거  ('할'이 2번 나옴)
    vocab = list(set(tokens))

    vocab_size = len(vocab)
    # len(vocab) = 42 ==> 유일한 단어가 42개

    # enumerate함수는 리스트의 원소에 순서값을 부여해주는 함수입니다
    # enumerate : 열거하다. 낱낱이 세다

    # >>> item = ["First", "Second", "Third"]
    # >>> for i, val in enumerate(item):
    # ... 	print("{} 번쨰 값은 {}입니다".format(i, val))
    #
    # 0 번쨰 값은 First입니다
    # 1 번쨰 값은 Second입니다
    # 2 번쨰 값은 Third입니다

    # for i, word in enumerate(vocab):
    #     print('word_to_ix1 = ', i, word)

    Word_to_ix = {Word: i for i, Word in enumerate(vocab)}
    ix_to_Word = {i: Word for i, Word in enumerate(vocab)}

    return tokens, vocab_size, Word_to_ix, ix_to_Word

################################################################################
# 활성화 함수
################################################################################

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def sigmoid_derivative(input):
    return input * (1 - input)

def tanh(input, derivative=False):
    return np.tanh(input)

def tanh_derivative(input):
    return 1 - input ** 2

def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))

################################################################################
# input_size=vocab_size + hidden_size, hidden_size=hidden_size, output_size=vocab_size
################################################################################
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, num_epochs,
                 learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        # Forget Gate
        self.Wf = np.random.randn(hidden_size, input_size) * 0.1
        self.bf = np.zeros((hidden_size, 1))
        print('class LSTM ==> self.Wf.shape = ', self.Wf.shape)
        # class LSTM == > self.Wf.shape = (25, 67)

        print('class LSTM ==> self.bf.shape = ', self.bf.shape)
        # class LSTM == > self.bf.shape = (25, 1)

        # Input Gate
        self.Wi = np.random.randn(hidden_size, input_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))

        # Candidate Gate
        self.Wc = np.random.randn(hidden_size, input_size) * 0.1
        self.bc = np.zeros((hidden_size, 1))

        # Output Gate
        self.Wo = np.random.randn(hidden_size, input_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))

        # Final Gate
        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    # 네트워크 메모리 리셋
    def reset(self):
        self.X = {}

        self.HS = {-1: np.zeros((self.hidden_size, 1))}
        self.CS = {-1: np.zeros((self.hidden_size, 1))}

        self.C = {}
        self.O = {}
        self.F = {}
        self.I = {}
        self.outputs = {}

    # Forward 순전파
    def forward(self, inputs):
        # self.reset()
        x = {}
        outputs = []
        for t in range(len(inputs)):  # 길이 42
            x[t] = np.zeros((vocab_size, 1))
            x[t][inputs[t]] = 1  # 각각의 Word에 대한 one hot coding
            #print('forward(self, inputs): => x[t] = \n', x[t])

            self.X[t] = np.concatenate((self.HS[t - 1], x[t]))
            print('forward(self, inputs): => self.HS[t - 1].shape = \n',self.HS[t - 1].shape)
            # forward(self, inputs): = > self.HS[t - 1].shape =  (25, 1)

            self.F[t] = sigmoid(np.dot(self.Wf, self.X[t]) + self.bf) # 망각 게이트
            self.I[t] = sigmoid(np.dot(self.Wi, self.X[t]) + self.bi) # 입력 게이트
            self.C[t] = tanh(np.dot(self.Wc, self.X[t]) + self.bc)    # 후보자 게이트
            self.O[t] = sigmoid(np.dot(self.Wo, self.X[t]) + self.bo) # 출력 게이트

            self.CS[t] = self.F[t] * self.CS[t - 1] + self.I[t] * self.C[t] # 상태 게이트
            self.HS[t] = self.O[t] * tanh(self.CS[t])

            outputs += [np.dot(self.Wy, self.HS[t]) + self.by]

        return outputs

    # 역전파
    def backward(self, errors, inputs):
        dLdWf, dLdbf = 0, 0
        dLdWi, dLdbi = 0, 0
        dLdWc, dLdbc = 0, 0
        dLdWo, dLdbo = 0, 0
        dLdWy, dLdby = 0, 0

        dh_next, dc_next = np.zeros_like(self.HS[0]), np.zeros_like(self.CS[0])
        for t in reversed(range(len(inputs))):
            error = errors[t]

            # Final Gate Weights and Biases Errors
            dLdWy += np.dot(error, self.HS[t].T)  # 𝜕𝐿/𝜕𝑊𝑦
            dLdby += error  # 𝜕𝐿/𝜕b𝑦 = (𝜕𝐿/𝜕z_t)(𝜕z_t/𝜕b𝑦) = error x 1 (Zt = WyHSt + by)

            # Hidden State Error
            dLdHS = np.dot(self.Wy.T, error) + dh_next  # 𝜕𝐿/𝜕𝐻𝑆

            # Output Gate Weights and Biases Errors
            dLdo = tanh(self.CS[t]) * dLdHS * sigmoid_derivative(self.O[t])
            dLdWo += np.dot(dLdo, inputs[t].T)
            dLdbo += dLdo

            # Cell State Error
            dLdCS = tanh_derivative(tanh(self.CS[t])) * self.O[
                t] * dLdHS + dc_next

            # Forget Gate Weights and Biases Errors
            dLdf = dLdCS * self.CS[t - 1] * sigmoid_derivative(self.F[t])
            dLdWf += np.dot(dLdf, inputs[t].T)
            dLdbf += dLdf

            # Input Gate Weights and Biases Errors
            dLdi = dLdCS * self.C[t] * sigmoid_derivative(self.I[t])
            dLdWi += np.dot(dLdi, inputs[t].T)
            dLdbi += dLdi

            # Candidate Gate Weights and Biases Errors
            dLdc = dLdCS * self.I[t] * tanh_derivative(self.C[t])
            dLdWc += np.dot(dLdc, inputs[t].T)
            dLdbc += dLdc

            # Concatenated Input Error (Sum of Error at Each Gate!)
            d_z = np.dot(self.Wf.T, dLdf) + np.dot(self.Wi.T, dLdi) + np.dot(
                self.Wc.T, dLdc) + np.dot(self.Wo.T, dLdo)

            # Error of Hidden State and Cell State at Next Time Step
            dh_next = d_z[:self.hidden_size, :]
            dc_next = self.F[t] * dLdCS

        for d_ in (
        dLdWf, dLdbf, dLdWi, dLdbi, dLdWc, dLdbc, dLdWo, dLdbo, dLdWy, dLdby):
            np.clip(d_, -1, 1, out=d_)

        self.Wf += dLdWf * self.learning_rate * (-1)
        self.bf += dLdbf * self.learning_rate * (-1)

        self.Wi += dLdWi * self.learning_rate * (-1)
        self.bi += dLdbi * self.learning_rate * (-1)

        self.Wc += dLdWc * self.learning_rate * (-1)
        self.bc += dLdbc * self.learning_rate * (-1)

        self.Wo += dLdWo * self.learning_rate * (-1)
        self.bo += dLdbo * self.learning_rate * (-1)

        self.Wy += dLdWy * self.learning_rate * (-1)
        self.by += dLdby * self.learning_rate * (-1)

    # Train
    def train(self, inputs, labels):
        for _ in tqdm(range(self.num_epochs)):
            self.reset()
            input_idx = [Word_to_ix[input] for input in inputs]
            # print('def train => input_idx = \n', input_idx)
            # def train => input_idx =
            # [12, 6, 32, 20, 4, 17, 38, 41, 40, 19, 30, 2, 8, 14, 34, 27, 22, 23,
            #  21, 16, 26, 3, 39, 31, 35, 25, 28, 13, 29, 10, 36, 7, 0, 1, 24, 5,
            #  33, 9, 11, 18, 37, 14]
            # print('def train => len(input_idx) = \n', len(input_idx))
            # def train => len(input_idx) = 42

            predictions = self.forward(input_idx)

            errors = []
            for t in range(len(predictions)):
                errors += [softmax(predictions[t])]
                errors[-1][Word_to_ix[labels[t]]] -= 1

            self.backward(errors, self.X)

    def test(self, inputs, labels):
        accuracy = 0
        probabilities = self.forward([Word_to_ix[input] for input in inputs])

        gt = ''
        output = '나라의 '

        for q in range(len(labels)):
            prediction = ix_to_Word[
                np.argmax(softmax(probabilities[q].reshape(-1)))]
            gt += inputs[q] + ' '
            output += prediction + ' '

            if prediction == labels[q]:
                accuracy += 1

        print('실제값: ', gt)
        print('예측값: ', output)

hidden_size = 25


# data preparation
tokens, vocab_size, Word_to_ix, ix_to_Word = data_preprocessing(data)
train_X, train_y = tokens[:-1], tokens[1:]
print('train_X = ',train_X)
print('train_y = ',train_y)
# train_X =  ['나라의', '말이', '중국과', '달라', '문자와', '서로', '통하지', '아니하기에', '이런', '까닭으로', '어리석은', '백성이', '이르고자', '할', '바가', '있어도', '마침내', '제', '뜻을', '능히', '펴지', '못할', '사람이', '많으니라', '내가', '이를', '위해', '가엾이', '여겨', '새로', '스물여덟', '글자를', '만드노니', '사람마다', '하여', '쉬이', '익혀', '날로', '씀에', '편안케', '하고자', '할']
# train_y =  ['말이', '중국과', '달라', '문자와', '서로', '통하지', '아니하기에', '이런', '까닭으로', '어리석은', '백성이', '이르고자', '할', '바가', '있어도', '마침내', '제', '뜻을', '능히', '펴지', '못할', '사람이', '많으니라', '내가', '이를', '위해', '가엾이', '여겨', '새로', '스물여덟', '글자를', '만드노니', '사람마다', '하여', '쉬이', '익혀', '날로', '씀에', '편안케', '하고자', '할', '따름이니라']

lstm = LSTM(input_size=vocab_size + hidden_size, hidden_size=hidden_size, output_size=vocab_size, num_epochs=1000,
            learning_rate=0.05)

##### Training #####
lstm.train(train_X, train_y)

lstm.test(train_X, train_y)
