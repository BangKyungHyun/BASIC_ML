
# 장기 의존성 문제란 입력과 출력 사이의 거리가 멀어질수록 연관 관계가 적어지는 문제
# 은닉층의 정보가 끝까지 전달되지 못하는 현상
# NumPy는 행렬이나 일반적으로 대규모 다차원 배열을 쉽게 처리할 수 있도록 지원하는 파이썬 라이브러리
import numpy as np
import re
import datetime

data = """
나라의 말이 중국과 달라 문자와 서로 통하지 아니하기에 이런 까닭으로 어리석은 백성이 이르고자 할 바가 있어도 마침내 제 뜻을 능히 펴지 못할 사람이 많으니라 내가 이를 위해 가엾이 여겨 새로 스물여덟 글자를 만드노니 사람마다 하여 쉬이 익혀 날로 씀에 편안케 하고자 할 따름이니라
"""

def data_preprocessing(data):
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

    data = re.sub('[^가-힣]', ' ', data)
    print('data = ', data)
    # data =   나라의 말이 중국과 달라 문자와 서로 통하지 아니하기에 이런 까닭으로 어리석은 백성이 이르고자 할 바가 있어도 마침내 제 뜻을 능히 펴지 못할 사람이 많으니라 내가 이를 위해 가엾이 여겨 새로 스물여덟 글자를 만드노니 사람마다 하여 쉬이 익혀 날로 씀에 편안케 하고자 할 따름이니라

    tokens = data.split()
    print('tokens = ', tokens)
    # tokens =  ['나라의', '말이', '중국과', '달라', '문자와', '서로', '통하지', '아니하기에', '이런', '까닭으로', '어리석은', '백성이', '이르고자', '할', '바가', '있어도', '마침내', '제', '뜻을', '능히', '펴지', '못할', '사람이', '많으니라', '내가', '이를', '위해', '가엾이', '여겨', '새로', '스물여덟', '글자를', '만드노니', '사람마다', '하여', '쉬이', '익혀', '날로', '씀에', '편안케', '하고자', '할', '따름이니라']

    #  list(set())을 이용한 중복제거  ('할'이 2번 나옴)
    vocab = list(set(tokens))
    print('list(set(tokens)) = ', list(set(tokens)))
    # list(set(tokens)) =  ['익혀', '문자와', '따름이니라', '날로', '이를', '까닭으로', '스물여덟', '사람마다', '아니하기에', '만드노니', '뜻을', '펴지', '못할', '글자를', '많으니라', '새로', '서로', '씀에', '편안케', '하여', '달라', '이르고자', '할', '위해', '가엾이', '바가', '백성이', '여겨', '쉬이', '중국과', '능히', '말이', '마침내', '이런', '통하지', '나라의', '제', '사람이', '하고자', '어리석은', '있어도', '내가']

    vocab_size = len(vocab)
    print('len(vocab) = ', len(vocab))
    #len(vocab) = 42 ==> 유일한 단어가 42개

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

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    print('word_to_ix = ', word_to_ix)
    # word_to_ix =  {'익혀': 0, '문자와': 1, '따름이니라': 2, '날로': 3, '이를': 4, '까닭으로': 5, '스물여덟': 6, '사람마다': 7, '아니하기에': 8, '만드노니': 9, '뜻을': 10, '펴지': 11, '못할': 12, '글자를': 13, '많으니라': 14, '새로': 15, '서로': 16, '씀에': 17, '편안케': 18, '하여': 19, '달라': 20, '이르고자': 21, '할': 22, '위해': 23, '가엾이': 24, '바가': 25, '백성이': 26, '여겨': 27, '쉬이': 28, '중국과': 29, '능히': 30, '말이': 31, '마침내': 32, '이런': 33, '통하지': 34, '나라의': 35, '제': 36, '사람이': 37, '하고자': 38, '어리석은': 39, '있어도': 40, '내가': 41}

    ix_to_word = {i: word for i, word in enumerate(vocab)}
    print('ix_to_word = ', ix_to_word)
    # ix_to_word =  {0: '익혀', 1: '문자와', 2: '따름이니라', 3: '날로', 4: '이를', 5: '까닭으로', 6: '스물여덟', 7: '사람마다', 8: '아니하

    return tokens, vocab_size, word_to_ix, ix_to_word

def init_weights(h_size, vocab_size):

    # print('def init_weights h_size =',h_size)
    # print('def init_weights vocab_size =',vocab_size)
    # def init_weights h_size = 100
    # def init_weights vocab_size = 42

    U = np.random.randn(h_size, vocab_size) * 0.01 # 100*42  input weight
    W = np.random.randn(h_size, h_size) * 0.01     # 100*100 hidden weight
    V = np.random.randn(vocab_size, h_size) * 0.01 # 42*100  h1 weight

    # print('def init_weights len(U) =',len(U))
    # print('def init_weights len(W) =',len(W))
    # print('def init_weights len(V) =',len(V))

    # def init_weights len(U) = 100
    # def init_weights len(W) = 100
    # def init_weights len(V) = 42

    return U,W,V

def feedforward(inputs, targets, hprev):

    loss = 0
    # {} 딕셔너리{Dictionary}
    # - 딕셔너리 {Dictionary}는 사전형태의 자료구조이다
    # - Key 와 Value으로 구분되며, Key를 사용하여 값에 접근 가능하다.
    # - 배열이나 튜플처럼 인덱스를 활용하여 Value 에 접근할 수 없다.

    # 선언문   : dic = {}
    # 초기화   : dic = {"january":1, "February": 2, "March":3 }
    # 불러오기 : dic["March"]

    xs, hs, os, ys = {}, {}, {}, {}

    # print('feedforward xs =',xs)
    # feedforward xs = {}

    # print('feedforward hs =',hs)
    # feedforward hs = {}

    # print('feedforward os =',os)
    # feedforward os = {}

    # 주어진 객체의 배열 복사본을 반환합니다.(이전 상태를 복사한다)
    hs[-1] = np.copy(hprev)
    #print('feedforward hs[-1] =',hs[-1])
    # hprev size는 100이며 0으로 초기화됨

    # print('feedforward => xs = ', xs)
    # feedforward = > xs = {}
    # print('feedforward => [inputs] = ', [inputs])
    # feedforward = > [inputs] = [[29, 21, 5, 28]]

    for i in range(seq_len):   # seq_len = 3

        # print('feedforward => i = ', i)
        # xs[i] = 입력 값
        xs[i] = np.zeros((vocab_size, 1)) # vocab_size = 42
        #print('feedforward => xs[i] 0으로 변환 = ', xs[i])
        # feedforward = > xs[i] 0 으로 변환 =
        # [[0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #  [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #  [0.][0.]]

        #print('feedforward => len(xs[i]) = ', len(xs[i]))
        # feedforward = > len(xs[i]) = 42

        # print('feedforward => [inputs[i]] = ', [inputs[i]])
        # feedforward = > [inputs[i]] = [16]
        # 나라의 인덱스가 16임

        xs[i][inputs[i]] = 1  # 각각의 word에 대한 one hot coding

        # print('feedforward => xs[i] = ', xs[i])
        #
        # feedforward = > xs[i] = [
        #     [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #     [1.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #     [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]]
        # print('feedforward => inputs[i]  = ', inputs[i])
        # feedforward => inputs[i]  =  33
        # print('feedforward => [inputs[i]]  = ', [inputs[i]])
        # feedforward => [inputs[i]]  =  [33]

        # print('feedforward => xs[i][inputs[i]] = ', xs[i][inputs[i]])
        # feedforward = > xs[i][inputs[i]] = [1.]
        # print('feedforward => len(xs[i][inputs[i]]) = ', len(xs[i][inputs[i]]))
        # feedforward = > len(xs[i][inputs[i]]) = 1

        # print('feedforward => U         = ', U)
        # print('feedforward => W         = ', W)
        # print('feedforward => hs[i - 1] = ', hs[i - 1])

        hs[i] = np.tanh(np.dot(U, xs[i]) + np.dot(W, hs[i - 1]))
        # print('feedforward => hs[i]     = ', hs[i])

        os[i] = np.dot(V, hs[i])
        # print('feedforward => os[i]     = ', os[i])

        ys[i] = np.exp(os[i]) / np.sum(np.exp(os[i]))  # softmax계산
        # print('feedforward => ys[i]     = ', ys[i])

        loss += -np.log(ys[i][targets[i], 0])          # loss 계산
        # print('feedforward => loss     = ', loss)

    return loss, ys, hs, xs

def backward(ys, hs, xs):

    # Backward propagation through time (BPTT)
    # 처음에 모든 가중치들은 0으로 설정
    dV = np.zeros(V.shape)
    dW = np.zeros(W.shape)
    dU = np.zeros(U.shape)

    for i in range(seq_len)[::-1]:

        # print('backward => i = ', i)
        output = np.zeros((vocab_size, 1)) # vocab_size = 42
        output[targets[i]] = 1
        # print('backward => output[targets[i]] = ', output[targets[i]])


        # reshape(-1,정수) : 위치에 -1인 경우
        # x.reshape(-1,1)
        # >>> array([[ 0],
        #        	   [ 1],
        #        	   [ 2],
        #            [ 3],
        #            [ 4],
        #            [ 5],
        #            [ 6],
        #            [ 7],
        #            [ 8],
        #            [ 9],
        #            [10],
        #            [11]])
        # x.reshape(-1,2)
        # >>> array([[ 0,  1],
        #            [ 2,  3],
        #            [ 4,  5],
        #            [ 6,  7],
        #            [ 8,  9],
        #            [10, 11]])
        # x.reshape(-1,3)
        # >>> array([[ 0,  1,  2],
        #            [ 3,  4,  5],
        #            [ 6,  7,  8],
        #            [ 9, 10, 11]])
        # 즉, 행(행)의 위치에 -1을 대신할 열의 값을 가집니다.

        # print('backward => output.reshape(-1, 1) = ', output.reshape(-1, 1))
        # print('backward => ys[i] 1 = ', ys[i])

        ys[i] = ys[i] - output.reshape(-1, 1)

        # print('backward => ys[i] 2 = ', ys[i])
        #
        # print('backward => (hs[i]) = ', (hs[i]))

        # 배열 전치하는 법 (T 메소드)
        #
        # A=np.array([[1,2,3],[4,5,6]])
        #
        # 2행 3열인 배열입니다.
        #
        # >>> A
        #
        # array([[1, 2, 3],
        #        [4, 5, 6]])
        #
        # 위 배열의 전치배열을 만들 때는 T 메소드를 사용합니다.
        #
        # >>> A.T
        #
        # array([[1, 4],
        #        [2, 5],
        #        [3, 6]])

        # 매번 i스텝에서 dL/dVi를 구하기  @는 나머지 계산
        # print('backward => (hs[i]).T = ', (hs[i]).T)
        # print('backward => ys[i] = ', ys[i])
        dV_step_i = ys[i] @ (hs[i]).T  # (y_hat - y) @ hs.T - for each step
        # print('backward => (dV_step_i = ', dV_step_i)

        dV = dV + dV_step_i  # dL/dVi를 다 더하기

        # 각i별로 V와 W를 구하기 위해서는
        # 먼저 공통적으로 계산되는 부분을 delta로 해서 계산해두고
        # 그리고 시간을 거슬러 dL/dWij와 dL/dUij를 구한 뒤
        # 각각을 합하여 dL/dW와 dL/dU를 구하고
        # 다시 공통적으로 계산되는 delta를 업데이트

        # i번째 스텝에서 공통적으로 사용될 delta
        delta_recent = (V.T @ ys[i]) * (1 - hs[i] ** 2)

        # 시간을 거슬러 올라가서 dL/dW와 dL/dU를 구하
        for j in range(i + 1)[::-1]:

            dW_ij = delta_recent @ hs[j - 1].T

            dW = dW + dW_ij

            dU_ij = delta_recent @ xs[j].reshape(1, -1)
            dU = dU + dU_ij

            # 그리고 다음번 j번째 타임에서 공통적으로 계산할 delta를 업데이트
            delta_recent = (W.T @ delta_recent) * (1 - hs[j - 1] ** 2)

        for d in [dU, dW, dV]:
            np.clip(d, -1, 1, out=d)

    return dU, dW, dV, hs[len(inputs) - 1]


def predict(word, length):

    print('predict => word = ', word)
    print('predict => length = ', length)
    x = np.zeros((vocab_size, 1)) # vocab_size = 42
    x[word_to_ix[word]] = 1
    print('predict => word_to_ix[word] = ', word_to_ix[word])

    ixes = []
    h = np.zeros((h_size,1)) # h_size = 100

    for t in range(length):

        print('predict => t ', t)
        # 입력 단어 x에 대한 상태값
        h = np.tanh(np.dot(U, x) + np.dot(W, h))
        # print('predict => x = ', x)

        y = np.dot(V, h)                     # 출력값 y hat
        p = np.exp(y) / np.sum(np.exp(y))    # 소프트맥스

        # 출력 단어의 인덱스 : 가장 높은 확률의 index를 리턴
        ix = np.argmax(p)

        print('predict => ix = ', ix)

        x = np.zeros((vocab_size, 1))        # 다음번 input x를 준비
        # print('predict => x = ', x)

        # 출력 단어(인덱스)가 다음 단계 반복문의 입력 단어(인덱스)가 됨
        # 나라의 말씀이 중국과 달라
        # 1번째 입력 단어 : 말씀이,  1번째 출력 단어 : 중국과
        # 2번째 입력 단어 : 중국과,  2번째 출력 단어 : 달라
        x[ix] = 1

        # 42번 출력값에 저장
        ixes.append(ix)
        print('predict => ixes = ', ixes)

    pred_words = ' '.join(ix_to_word[i] for i in ixes)

    print('predict => pred_words = ', pred_words)

    return pred_words

# 기본적인 parameters
epochs = 100000
h_size = 100
seq_len = 4
learning_rate = 1e-3

tokens, vocab_size, word_to_ix, ix_to_word = data_preprocessing(data)

U, W, V = init_weights(h_size, vocab_size)

p = 0

# h_size 행 , 1열을 가진 배열 생성
hprev = np.zeros((h_size, 1))
print('len(hprev) =',len(hprev))
print('main len(tokens) =',len(tokens))
# main len(tokens) = 43  # 43개의 단어, 중복을 제거하면 42개 단어

for epoch in range(epochs):

    for p in range(len(tokens)-seq_len):  # 43-3 = 40

        # print('p =, p + seq_len ', p, p + seq_len)

        inputs = [word_to_ix[tok] for tok in tokens[p:p + seq_len]]
        # print('tokens[p:p + seq_len] = ',tokens[p:p + seq_len])
        # print('[word_to_ix[tok] = ',[word_to_ix['나라의']])
        # print('inputs = ', inputs)

        targets = [word_to_ix[tok] for tok in tokens[p + 1:p + seq_len + 1]]
        # print('tokens[p + 1:p + seq_len + 1] = ',tokens[p + 1:p + seq_len + 1])
        # print('targets = ', targets)

        loss, ys, hs, xs = feedforward(inputs, targets, hprev)

        dU, dW, dV, hprev = backward(ys, hs, xs)

        # Update weights and biases using gradient descent
        W -= learning_rate * dW
        U -= learning_rate * dU
        V -= learning_rate * dV

        # p += seq_len

    if epoch % 1000 == 0:
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime,f'epoch {epoch}, loss: {loss}')

while 1:
    try:
        user_input = input("input: ")
        if user_input == 'break':
            break
        response = predict(user_input,40)
        print(response)
    except:
        print('Uh oh try again!')

# dictionary test

# dic = {}
#
# dic = {"january":1, "February": 2, "March":3 }
# print('dic 1 =', dic)
# # dic 1 = {'january': 1, 'February': 2, 'March': 3}
#
# dic = 0
# print('dic 2 =', dic)
# # dic 2 = 0
#
# dic = np.zeros((3, 1))
# print('dic 3 =', dic)
# # dic 3 = [[0.] [0.] [0.]]
#
# dic = np.zeros((3, 2))
# print('dic 4 =', dic)
# # dic 4 = [[0. 0.] [0. 0.] [0. 0.]]
#
# dic = np.zeros((3, 3, 2))
# print('dic 5 =', dic)
# dic 5 = [[[0. 0.]   [0. 0.]   [0. 0.]]
#          [[0. 0.]   [0. 0.]   [0. 0.]]
#          [[0. 0.]   [0. 0.]   [0. 0.]]]
