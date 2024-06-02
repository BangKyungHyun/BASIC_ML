from typing import List, Any

from tqdm import tqdm  # 반복문에서 진행률을 보여주는 기능 수행
import numpy as np
import re

##### Data #####
data = "나라의 말이 중국과 달라 문자와 서로 통하지 아니하기에 이런 까닭으로 어리석은 백성이 이르고자 할 바가 있어도 마침내 제 뜻을 능히 펴지 못할 사람이 많으니라 내가 이를 위해 가엾이 여겨 새로 스물여덟 글자를 만드노니 사람마다 하여 쉬이 익혀 날로 씀에 편안케 하고자 할 따름이니라"

################################################################################
#데이터를 preprocessing 해주는 부분
################################################################################
# re.sub(pattern, replace, text) : text 중 pattern에 해당하는 부분을 replace로 대체한다.
# data = re.sub('[^가-힣]', ' ', data): 자모만 아닌 한글만 남기기(공백 제거)
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
    print('data =', data)
    # data = 나라의 말이 중국과 달라 문자와 서로 통하지 아니하기에 이런 까닭으로 어리석은 백성이 이르고자 할 바가 있어도 마침내 제 뜻을 능히 펴지 못할 사람이 많으니라 내가 이를 위해 가엾이 여겨 새로 스물여덟 글자를 만드노니 사람마다 하여 쉬이 익혀 날로 씀에 편안케 하고자 할 따름이니라

    # data.split() -문자열을 쪼개주는 함수. 공백을 넣을 경우 띄어쓰기 기준으로 분리. 분리된 문자열을 리스트의 원소로 저장
    tokens = data.split()
    print('tokens =', tokens)
    print('tokens size = ',len(tokens))
    # tokens = ['나라의', '말이', '중국과', '달라', '문자와', '서로', '통하지', '아니하기에', '이런', '까닭으로', '어리석은', '백성이', '이르고자', '할', '바가', '있어도', '마침내', '제', '뜻을', '능히', '펴지', '못할', '사람이', '많으니라', '내가', '이를', '위해', '가엾이', '여겨', '새로', '스물여덟', '글자를', '만드노니', '사람마다', '하여', '쉬이', '익혀', '날로', '씀에', '편안케', '하고자', '할', '따름이니라']
    # tokens size = 43

    #  list(set())을 이용한 중복제거  ('할'이 2번 나옴)
    vocab = list(set(tokens))

    vocab_size = len(vocab)
    # len(vocab) = 42 ==> 유일한 단어가 42개

    # enumerate함수는 리스트의 원소에 순서값을 부여해 주는 함수
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
    print('word_to_ix = ', Word_to_ix)
    # word_to_ix =  {'말이': 0, '통하지': 1, '쉬이': 2, '하고자': 3, '따름이니라': 4, '씀에': 5, '제': 6, '중국과': 7, '문자와': 8, '백성이': 9,
    # '능히': 10, '서로': 11, '날로': 12, '많으니라': 13, '사람마다': 14, '새로': 15, '할': 16, '나라의': 17, '못할': 18, '만드노니': 19,
    # '여겨': 20, '마침내': 21, '어리석은': 22, '사람이': 23, '내가': 24, '까닭으로': 25, '이르고자': 26, '아니하기에': 27, '뜻을': 28, '가엾이': 29,
    # '달라': 30, '하여': 31, '스물여덟': 32, '익혀': 33, '펴지': 34, '위해': 35, '글자를': 36, '편안케': 37, '이를': 38, '바가': 39, '이런': 40,
    # '있어도': 41}

    ix_to_Word = {i: Word for i, Word in enumerate(vocab)}
    print('ix_to_word = ', ix_to_Word)
    # ix_to_word =  {0: '말이', 1: '통하지', 2: '쉬이', 3: '하고자', 4: '따름이니라', 5: '씀에', 6: '제', 7: '중국과', 8: '문자와', 9: '백성이',
    # 10: '능히', 11: '서로', 12: '날로', 13: '많으니라', 14: '사람마다', 15: '새로', 16: '할', 17: '나라의', 18: '못할', 19: '만드노니',
    # 20: '여겨', 21: '마침내', 22: '어리석은', 23: '사람이', 24: '내가', 25: '까닭으로', 26: '이르고자', 27: '아니하기에', 28: '뜻을',
    # 29: '가엾이', 30: '달라', 31: '하여', 32: '스물여덟', 33: '익혀', 34: '펴지', 35: '위해', 36: '글자를', 37: '편안케', 38: '이를', 39: '바가',
    # 40: '이런', 41: '있어도'}

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
# 67 = 42+25                                       25                     42
################################################################################

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        # Forget Gate
        self.Wf = np.random.randn(hidden_size, input_size) * 0.1
        self.bf = np.zeros((hidden_size, 1))
        # print('class LSTM ==> self.Wf.shape = ', self.Wf.shape)
        # class LSTM == > self.Wf.shape = (25, 67)
        # print('class LSTM ==> self.bf.shape = ', self.bf.shape)
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
        # print('class LSTM ==> self.Wy.shape = ', self.Wy.shape)
        # print('class LSTM ==> self.by.shape = ', self.by.shape)
        # class LSTM == > self.Wy.shape =  (42, 25)
        # class LSTM == > self.by.shape =  (42, 1)

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
    # 입력에 대한 추정 출력값을 계산(1에서 42번 문자까지 반복)
    # 입력 1  : 나라의  => 출력 1  : 말이
    # 입력 42 : 할     => 출력 42 : 따름이니라
    def forward(self, inputs):
        # self.reset()
        x = {}   # 각각의 Word에 대한 one hot coding용 변수

        # print('forward( inputs ) = ',  inputs)
        # forward_count = -1
        outputs = []
        for t in range(len(inputs)):  # 길이 42

            # forward_count += 1
            x[t] = np.zeros((vocab_size, 1))

            x[t][inputs[t]] = 1  # 각각의 Word에 대한 one hot coding
            # print('forward(self, inputs): => x[t].shape = \n',x[t].shape)
            # forward(self, inputs): = > x[t].shape = (42, 1)

            # 첫번째 단어인 경우 프린트
            # if t <= 3:                # and forward_count <= 1:
            # print('forward(t,forward_count,  x[t] = ', t, forward_count,  inputs)

            # print('forward(self, inputs): => x[t] = \n', x[t])
            # x[t] =
            # [[0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.]
            #  [1.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.]
            #  [0.] [0.] [0.] [0.]]

            self.X[t] = np.concatenate((self.HS[t - 1], x[t]))        # 입력 값(일종의 단기 기억) = 이전 상태+입력 단어(67개 생성)
            # print('forward(self, inputs): => self.HS[t - 1].shape = ',self.HS[t - 1].shape)  # 은닉층 갯수 만큼 생성 25개
            # # forward(self, inputs): = > self.HS[t - 1].shape = (25, 1)
            # print('forward(self, inputs): => ', self.X[t].shape)
            # forward(self, inputs): = > (67, 1)
            # forward(self, inputs): => self.HS[t - 1] =
            #  [[0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.] [0.]]
            # print('forward(self, inputs): => np.concatenate((self.HS[t - 1], x[t])) = \n',np.concatenate((self.HS[t - 1], x[t])))

            # forward(self, inputs): = > np.concatenate((self.HS[t - 1], x[t])) = 67개 생성
            self.F[t] = sigmoid(np.dot(self.Wf, self.X[t]) + self.bf) # 망각 게이트
            self.I[t] = sigmoid(np.dot(self.Wi, self.X[t]) + self.bi) # 입력 게이트
            self.C[t] = tanh(np.dot(self.Wc, self.X[t]) + self.bc)    # 후보자 게이트
            self.O[t] = sigmoid(np.dot(self.Wo, self.X[t]) + self.bo) # 출력 게이트

            # 현재 셀 상태 = 이전 셀 상태 * 망각게이트 + 후보자게이트 * 입력게이트
            self.CS[t] = self.CS[t - 1] * self.F[t] + self.I[t] * self.C[t]

            # 은닉상태 = 출력 게이트 * 셀 상태
            self.HS[t] = self.O[t] * tanh(self.CS[t])

            # 순전파의 outputs인 logit를 계산하는 부분 - 강의에서 Zt 값
            outputs += [np.dot(self.Wy, self.HS[t]) + self.by]
            # print('forward(self, inputs): => outputs = ', len(outputs))
            # forward(self, inputs): => outputs =  1
            # forward(self, inputs): => outputs =  2
            #    1~4까지 반복
            # forward(self, inputs): => outputs =  42

            # 출력 값 = 출력 가중치
            # print('forward(self, inputs): => self.HS[t - 1].shape = \n',self.HS[t - 1].shape)
            # print('forward(self, inputs): => self.F[t].shape = ', self.F[t].shape)
            # print('forward(self, inputs): => self.I[t].shape = ', self.I[t].shape)
            # print('forward(self, inputs): => self.C[t].shape = ', self.C[t].shape)
            # print('forward(self, inputs): => self.O[t].shape = ', self.O[t].shape)
            # print('forward(self, inputs): => self.CS[t].shape = ', self.CS[t].shape)
            # print('forward(self, inputs): => self.HS[t].shape = ', self.HS[t].shape)
            # forward(self, inputs): = > self.HS[t - 1].shape =  (25, 1)
            # forward(self, inputs): => self.F[t].shape =  (25, 1)
            # forward(self, inputs): => self.I[t].shape =  (25, 1)
            # forward(self, inputs): => self.C[t].shape =  (25, 1)
            # forward(self, inputs): => self.O[t].shape =  (25, 1)
            # forward(self, inputs): => self.CS[t].shape =  (25, 1)
            # forward(self, inputs): => self.HS[t].shape =  (25, 1)
            # print('--------------------------------------------------------')
            # print('forward(self, inputs): => x = \n', x)
            # print('forward(self, inputs): => self.CS = \n',self.CS)
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
            dLdCS = tanh_derivative(tanh(self.CS[t])) * self.O[t] * dLdHS + dc_next

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
    def train(self, inputs, labels):  # inputs : train_x 42개 입력 => 마지막 단어 '따름이니라' 제외된 상태
                                      # labels : train_y 42개 출력 => 첫번째 단어 '나라의' 제외된 상태

        for _ in tqdm(range(self.num_epochs),mininterval=0.1):  #1000번
            # print('def train => _) = ', _)

            self.reset()
            # 입력 42 단어(마지막 1단어 제외)를 인덱스 값으로 변환
            input_idx = [Word_to_ix[input] for input in inputs]

            # print('def train => inputs = \n', inputs)
            # # def inputs = ['나라의', '말이', '중국과', '달라', '문자와', '서로', '통하지', '아니하기에', '이런', '까닭으로', '어리석은', '백성이', '이르고자', '할', '바가', '있어도', '마침내', '제', '뜻을', '능히', '펴지', '못할', '사람이', '많으니라' '내가',
            # #             '이를', '위해', '가엾이', '여겨', '새로', '스물여덟', '글자를', '만드노니','사람마다', '하여', '쉬이', '익혀', '날로', '씀에', '편안케', '하고자', '할']

            # print('def train => input_idx = \n', input_idx)

            # def train => input_idx =
            # [17, 0, 7, 30, 8, 11, 1, 27, 40, 25, 22, 9, 26, 16, 39, 41, 21, 6, 28, 10, 34, 18, 23, 13, 24, 38, 35, 29, 20, 15, 32, 36, 19, 14, 31, 2, 33, 12, 5, 3
            # train_X =  ['나라의', '말이', '중국과', '달라', '문자와', '서로', '통하지', '아니하기에', '이런', '까닭으로', '어리석은', '백성이', '이르고자', '할', '바가', '있어도', '마침내', '제', '뜻을', '능히', '펴지', '못할', '사람이', '많으니라', '내가', '이를', '위해', '가엾이', '여겨', '새로', '스물여덟', '글자를', '만드노니', '사람마다', '하여', '쉬이', '익혀', '날로', '씀에', '편안케', '하고자', '할']

            # print('def train => len(input_idx) = \n', len(input_idx))
            # def train => len(input_idx) = 42



            predictions = self.forward(input_idx)
            # print('def train => seq, len(predictions) = ', _, len(predictions))
            # def train => seq, len(predictions) = 0 42
            print('def train => ,ephochs predictions[0] = ', _, predictions[0])
            # [[-0.05660152]         [0.01904128]         [-0.25867935]         [0.11951098]         [-0.04462367]         [-0.28279924]         [0.11561574]         [-0.16591567]         [0.22419834]
            #  [-0.02400536]         [0.13376789]         [-0.14749723]         [-0.28561582]        [-0.15672086]         [-0.11142088]         [-0.00065546]        [-0.20818995]         [-0.13196902]
            #  [-0.05695697]         [-0.22867049]        [-0.01356011]         [0.06840347]         [-0.0859211]          [0.14779235]          [-0.0909562]         [0.04386346]          [0.07618501]
            #  [0.17334264]          [0.33092089]         [-0.00355083]         [0.0013071]          [0.06720417]          [0.18258586]          [0.19363844]         [-0.18524848]         [0.18641261]
            #  [0.0605055]           [-0.1685856]         [0.16498516]          [-0.16153445]        [0.07364889]          [-0.22927805]]

            errors = []
            for t in range(len(predictions)):
                # output에 softmax 처리하면 y hat이 됨
                errors += [softmax(predictions[t])]
                # print('def train => errors = ', errors)
                # print('def train => errors.size = ', len(errors))

                # print('def train => errors[-1][Word_to_ix[labels[t]]] = ',
                #       errors[-1][Word_to_ix[labels[t]]])
                errors[-1][Word_to_ix[labels[t]]] -= 1
                # print('------------')
                # print('def train => errors[-1] = ', errors[-1])
                # print('def train => errors[-1][Word_to_ix[labels[t]]] = ',
                #       errors[-1][Word_to_ix[labels[t]]])
                # print('def train => Word_to_ix[labels[t]] = ', Word_to_ix[labels[t]])

            # print('def train => len(errors) = ', len(errors))
            # def train => len(errors) = 42
            # if _ % 10000 == 0:
            #     print('def train => errors[41] = ', _, errors[41].reshape(-1))

            self.backward(errors, self.X)  # 예측값과 실제값

    def test(self, inputs, labels):

        accuracy = 0
        # 입력값에 대해 학습된 가중치를 바탕으로 출력값(확률)을 산출 -> 가중치가 결정되어 역전파는 필요 없음
        probabilities = self.forward([Word_to_ix[input] for input in inputs])

        # print('def test => len(probabilities) = ', len(probabilities))
        # def test => len(probabilities) = 42

        # print('def test => probabilities[0] = \n', probabilities[0])
        # def test => probabilities[0] =
        # [[-0.05660152]         [0.01904128]         [-0.25867935]         [0.11951098]         [-0.04462367]         [-0.28279924]         [0.11561574]         [-0.16591567]         [0.22419834]
        #  [-0.02400536]         [0.13376789]         [-0.14749723]         [-0.28561582]        [-0.15672086]         [-0.11142088]         [-0.00065546]        [-0.20818995]         [-0.13196902]
        #  [-0.05695697]         [-0.22867049]        [-0.01356011]         [0.06840347]         [-0.0859211]          [0.14779235]          [-0.0909562]         [0.04386346]          [0.07618501]
        #  [0.17334264]          [0.33092089]         [-0.00355083]         [0.0013071]          [0.06720417]          [0.18258586]          [0.19363844]         [-0.18524848]         [0.18641261]
        #  [0.0605055]           [-0.1685856]         [0.16498516]          [-0.16153445]        [0.07364889]          [-0.22927805]]

        # print('def test => softmax(probabilities[0].reshape(-1)) = \n', softmax(probabilities[0].reshape(-1)))
        # def test => softmax(probabilities[0].reshape(-1)) =
        # [0.02109976 0.02285346 0.02642524 0.02298963 0.02233014 0.02237085 0.02078446 0.02461876 0.01985594 0.01636164 0.02051823 0.0227193  0.02886429 0.02170379 0.02788705 0.02748051 0.02196338 0.02745894
        #  0.02901851 0.0220207  0.01748547 0.0210674  0.02196035 0.02500596 0.0256519  0.0244947  0.02864787 0.02457471 0.02872344 0.02284624 0.02739873 0.02045484 0.02623577 0.02204887 0.02881358 0.02284396
        #  0.02071633 0.02128068 0.02436988 0.02108462 0.02695595 0.02801413]

        # print('def test => np.argmax(softmax(probabilities[0].reshape(-1))) = ', np.argmax(softmax(probabilities[0].reshape(-1))))
        # def test => np.argmax(softmax(probabilities[0].reshape(-1))) = 18

        # print('def test => ix_to_Word[np.argmax(softmax(probabilities[0].reshape(-1)))] = ', ix_to_Word[np.argmax(softmax(probabilities[0].reshape(-1)))])
        # def test => ix_to_Word[np.argmax(softmax(probabilities[0].reshape(-1)))] = 내가


        gt = ''
        output = '나라의 '

        for q in range(len(labels)):   # labels : 실제 결과 값
            prediction = ix_to_Word[np.argmax(softmax(probabilities[q].reshape(-1)))]
            print('def test => prediction, labels[q] = ', prediction, labels[q])

            gt += inputs[q] + ' '    # input 값은 단순히 concatenate 작업 만 수행
            output += prediction + ' '

            if prediction == labels[q]:
                accuracy += 1

        print('실제값: ', gt)
        print('예측값: ', output)
        print('정확도: ', accuracy)

################################################################################
#   MAIN ROUTINE
################################################################################

# hidden_size 가 25인 이유는 뭘까?
hidden_size = 25

# data preparation
tokens, vocab_size, Word_to_ix, ix_to_Word = data_preprocessing(data)

train_X, train_y = tokens[:-1], tokens[1:]

# train_x 42개 입력 => 마지막 단어 '따름이니라' 제외된 상태  
# train_y 42개 출력 => 첫번째 단어 '나라의' 제외된 상태
# 예) 입력이 '나라의' 이면 출력이 '말이' 출력 되어야 함

print('train_X = ',train_X)
print('train_y = ',train_y)
# train_X =  ['나라의', '말이', '중국과', '달라', '문자와', '서로', '통하지', '아니하기에', '이런', '까닭으로', '어리석은', '백성이', '이르고자', '할', '바가', '있어도', '마침내', '제', '뜻을', '능히', '펴지', '못할', '사람이', '많으니라', '내가', '이를', '위해', '가엾이', '여겨', '새로', '스물여덟', '글자를', '만드노니', '사람마다', '하여', '쉬이', '익혀', '날로', '씀에', '편안케', '하고자', '할']
# train_y =  ['말이', '중국과', '달라', '문자와', '서로', '통하지', '아니하기에', '이런', '까닭으로', '어리석은', '백성이', '이르고자', '할', '바가', '있어도', '마침내', '제', '뜻을', '능히', '펴지', '못할', '사람이', '많으니라', '내가', '이를', '위해', '가엾이', '여겨', '새로', '스물여덟', '글자를', '만드노니', '사람마다', '하여', '쉬이', '익혀', '날로', '씀에', '편안케', '하고자', '할', '따름이니라']

# LSTM 객체 생성
lstm = LSTM(input_size=vocab_size + hidden_size, hidden_size=hidden_size, output_size=vocab_size, num_epochs=30,learning_rate=0.05)

##### Training #####
lstm.train(train_X, train_y)

lstm.test(train_X, train_y)


