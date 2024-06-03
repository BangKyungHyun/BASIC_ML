import numpy as np
import re

data = """
나라의 말이 중국과 달라 문자와 서로 통하지 아니하기에 이런 까닭으로 어리석은 백성이 이르고자 할 바가 있어도 마침내 제 뜻을 능히 펴지 못할 사람이 많으니라 내가 이를 위해 가엾이 여겨 새로 스물여덟 글자를 만드노니 사람마다 하여 쉬이 익혀 날로 씀에 편안케 하고자 할 따름이니라
"""

def data_preprocessing(data):
    data = re.sub('[^가-힣]', ' ', data)  # 띄어쓰기로 단위로 토큰을 생성
    tokens = data.split()
    #  list(set())을 이용한 중복제거  ('할'이 2번 나옴)
    vocab = list(set(tokens))
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}  # 인코딩 문자 -> 인덱스
    ix_to_word = {i: word for i, word in enumerate(vocab)}  # 디코딩 인덱스 -> 문자

    return tokens, vocab_size, word_to_ix, ix_to_word

def init_weights(h_size, vocab_size):

    U = np.random.randn(h_size, vocab_size) * 0.01  # 100*42   x1 가중치
    W = np.random.randn(h_size, h_size) * 0.01      # 100*100  h0 가중치
    V = np.random.randn(vocab_size, h_size) * 0.01  # 42*100   h1 가중치

    # print('init_weights U shape = ', U.shape)
    # print('init_weights W shape = ', W.shape)
    # print('init_weights V shape = ', V.shape)

    # init_weights  U shape = (100, 42)
    # init_weights  W shape = (100, 100)
    # init_weights  V shape = (42, 100)

    # print('init_weights U  = ', U)
    #
    # [[0.00866254 - 0.00345567  0.01051043...    0.00677608 - 0.0156563    0.00699676]
    #  [-0.01529971  0.00725647 - 0.0013497...  - 0.0031064  - 0.00179495 - 0.00144175]
    #  [0.00761764    0.00502334 - 0.01789428... - 0.00665269   0.01618736 - 0.00589379]
    #  ...
    #  [-0.00665883   0.01887875 - 0.00791347...   0.0237582  - 0.00564357   0.00872708]
    #  [-0.01004637  0.00907099  - 0.01513228...   0.00460009   0.01204605   0.01948734]
    #  [0.00414538 - 0.01077217    0.0108689...  - 0.00910713   0.01076645 - 0.00173939]]

    return U,W,V

def feedforward(inputs, targets, hprev):

    # print('feedforward inputs = ', inputs)
    # print('feedforward targets = ', targets)
    # print('feedforward hprev = ', hprev)

    # feedforward inputs = [39, 3, 11] 예) 나라의 말씀이 중국과
    # feedforward targets = [3, 11, 7] 예) 말씀이 중국과 달라
    # feedforward hprev = [[0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
    #                      [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
    #                      [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
    #                      [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
    #                      [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]]
    #                      예) [0.] 이 100개임

    loss = 0

    xs, hs, ps, ys = {}, {}, {}, {}

    # 이전 은닉값을 0를 100개로 초기화
    hs[-1] = np.copy(hprev)

    for i in range(seq_len):     # seq_len = 3

        # 입력값 초기화
        xs[i] = np.zeros((vocab_size, 1))   # vocab_size = 42  => [42,1]

        # xs[i] = [[0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #          [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #          [0.][0.]]

        # xs[i][0] = 9
        # xs[i][1] = 11
        # xs[i] = [[9.][11.][0.][0.][0.][0.][1.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #          [0.][0.] [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #          [0.][0.]]

        # print('feedforward inputs[i] =', inputs[i])
        # inputs[i] = 0    예) 나라의


        # 초기화된 입력값에 각각의 word에 대한 one hot coding
        xs[i][inputs[i]] = 1
        # print('feedforwardxs[i][inputs[i]] =', xs[i][inputs[i]])
        # feedforwardxs[i][inputs[i]] = [1.]

        # print('feedforward xs[i] =', xs[i])
        # xs[i] = [[1.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #          [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #          [0.][0.]]

        # print('def feedforward(inputs, targets, hprev): = ', i)
        # def feedforward(inputs, targets, hprev): = 0
        # def feedforward(inputs, targets, hprev): = 1
        # def feedforward(inputs, targets, hprev): = 2

        hs[i] = np.tanh(np.dot(U, xs[i]) + np.dot(W, hs[i - 1]))
        ys[i] = np.dot(V, hs[i])
        ps[i] = np.exp(ys[i]) / np.sum(np.exp(ys[i]))  # softmax계산
        loss += -np.log(ps[i][targets[i], 0])

    return loss, ps, hs, xs


def backward(ps, hs, xs):

    # Backward propagation through time (BPTT)
    # 처음에 모든 가중치들은 0으로 설정
    dV = np.zeros(V.shape)
    dW = np.zeros(W.shape)
    dU = np.zeros(U.shape)

    for i in range(seq_len)[::-1]:
        output = np.zeros((vocab_size, 1))
        output[targets[i]] = 1
        ps[i] = ps[i] - output.reshape(-1, 1)
        # 매번 i스텝에서 dL/dVi를 구하기
        dV_step_i = ps[i] @ (hs[i]).T  # (y_hat - y) @ hs.T - for each step

        dV = dV + dV_step_i  # dL/dVi를 다 더하기

        # 각i별로 V와 W를 구하기 위해서는
        # 먼저 공통적으로 계산되는 부분을 delta로 해서 계산해두고
        # 그리고 시간을 거슬러 dL/dWij와 dL/dUij를 구한 뒤
        # 각각을 합하여 dL/dW와 dL/dU를 구하고
        # 다시 공통적으로 계산되는 delta를 업데이트

        # i번째 스텝에서 공통적으로 사용될 delta
        delta_recent = (V.T @ ps[i]) * (1 - hs[i] ** 2)

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

#===============================================================================
#===============================================================================
def predict(word, length):

    x = np.zeros((vocab_size, 1))
    x[word_to_ix[word]] = 1    # 입력단어를 원핫인코딩 처리

    print('predict word =', word)
    print('predict length = ', length)
    print('predict x[word_to_ix[word]] =', x[word_to_ix[word]])
    print('predict h_size =', h_size)

    ixes = []
    h = np.zeros((h_size,1))

    for t in range(length):
        #print('predict x =', x)
        # x = [[1.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #      [0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.][0.]
        #      [0.][0.]]
        h = np.tanh(np.dot(U, x) + np.dot(W, h))
        y = np.dot(V, h)
        p = np.exp(y) / np.sum(np.exp(y))    # 소프트맥스

        ix = np.argmax(p)                    # 가장 높은 확률의 index를 리턴

        x = np.zeros((vocab_size, 1))        # 다음번 input x를 준비
        x[ix] = 1
        print('predict ix, x[ix] = ', ix, x[ix])

        ixes.append(ix)

    # predict ix, x[ix] =  30 [1.]
    # predict ix, x[ix] =  36 [1.]
    # predict ix, x[ix] =  25 [1.]
    # predict ix, x[ix] =  13 [1.]
    # predict ix, x[ix] =  29 [1.]
    # predict ix, x[ix] =  30 [1.]
    # predict ix, x[ix] =  36 [1.]
    # predict ix, x[ix] =  25 [1.]
    # predict ix, x[ix] =  13 [1.]
    # predict ix, x[ix] =  29 [1.]
    # predict ix, x[ix] =  30 [1.]
    # predict ix, x[ix] =  36 [1.]
    # predict ix, x[ix] =  25 [1.]
    # predict ix, x[ix] =  13 [1.]
    # predict ix, x[ix] =  29 [1.]
    # predict ix, x[ix] =  30 [1.]
    # predict ix, x[ix] =  36 [1.]
    # predict ix, x[ix] =  25 [1.]
    # predict ix, x[ix] =  13 [1.]
    # predict ix, x[ix] =  29 [1.]
    # predict ix, x[ix] =  30 [1.]
    # predict ix, x[ix] =  36 [1.]
    # predict ix, x[ix] =  25 [1.]
    # predict ix, x[ix] =  13 [1.]
    # predict ix, x[ix] =  29 [1.]
    # predict ix, x[ix] =  30 [1.]
    # predict ix, x[ix] =  36 [1.]
    # predict ix, x[ix] =  25 [1.]
    # predict ix, x[ix] =  13 [1.]
    # predict ix, x[ix] =  29 [1.]
    # predict ix, x[ix] =  30 [1.]
    # predict ix, x[ix] =  36 [1.]
    # predict ix, x[ix] =  25 [1.]
    # predict ix, x[ix] =  13 [1.]
    # predict ix, x[ix] =  29 [1.]
    # predict ix, x[ix] =  30 [1.]
    # predict ix, x[ix] =  36 [1.]
    # predict ix, x[ix] =  25 [1.]
    # predict ix, x[ix] =  13 [1.]
    # predict ix, x[ix] =  29 [1.]
    # predict ixes =  [30, 36, 25, 13, 29, 30, 36, 25, 13, 29, 30, 36, 25, 13, 29, 30, 36, 25, 13, 29, 30, 36, 25, 13, 29, 30, 36, 25, 13, 29, 30, 36, 25, 13, 29, 30, 36, 25, 13, 29]

    print('predict ixes = ', ixes)

    pred_words = ' '.join(ix_to_word[i] for i in ixes)

    # 까닭으로 바가 뜻을 능히 통하지 까닭으로 바가 뜻을 능히 통하지 까닭으로 바가 뜻을 능히 통하지 까닭으로 바가 뜻을 능히 통하지 까닭으로 바가 뜻을 능히 통하지 까닭으로 바가 뜻을 능히 통하지 까닭으로 바가 뜻을 능히 통하지 까닭으로 바가 뜻을 능히 통하지

    return pred_words

# 기본적인 parameters
epochs = 1
h_size = 100
seq_len = 3
learning_rate = 1e-2

tokens, vocab_size, word_to_ix, ix_to_word = data_preprocessing(data)

print('tokens =', tokens)
print('word_to_ix =', word_to_ix)
print('ix_to_word =', ix_to_word)
#                         100,    42
U, W, V = init_weights(h_size, vocab_size)

print('U.shape =',U.shape)
print('W.shape =',W.shape)
print('V.shape =',V.shape)
print('len(tokens) =', len(tokens))
# len(tokens) = 43
print('vocab_size =', vocab_size)
# vocab_size = 42

# len(U) = 100
# len(W) = 100
# len(V) = 42
p = 0
hprev = np.zeros((h_size, 1))
# print('hprev =',hprev)

for epoch in range(epochs):

    # count = 0
                   # 43-3 = 40
    for p in range(len(tokens)-seq_len):   # seq_len이 3이므로 전체 문자열(43)에서 3을 뺀 40번 만큼 반복함

        # print('len(tokens)-seq_len, p, p + seq_len  =', len(tokens)-seq_len, p, p + seq_len)
        # len(tokens) - seq_len, p, p + seq_len = 40 0 3
        # len(tokens) - seq_len, p, p + seq_len = 40 0 3
        # len(tokens) - seq_len, p, p + seq_len = 40 1 4
        # .............
        # len(tokens) - seq_len, p, p + seq_len = 40 37 40
        # len(tokens) - seq_len, p, p + seq_len = 40 38 41
        # len(tokens) - seq_len, p, p + seq_len = 40 39 42

        inputs = [word_to_ix[tok] for tok in tokens[p:p + seq_len]]

        # token_seq = tokens[p]
        # print('[word_to_ix[token_seq]] = ', [word_to_ix[token_seq]])
        # [word_to_ix[token_seq]] = [26]  예) 나라의
        # [word_to_ix[token_seq]] = [23]  예) 말이
        # .....
        # [word_to_ix[token_seq]] = [23]  예) 편안케

        # print('inputs =', inputs)
        # inputs = [18, 23, 21] 예) 나라의 말씀이 중국과

        targets = [word_to_ix[tok] for tok in tokens[p + 1:p + seq_len + 1]]
        # print('targets =', targets)
        # targets = [23, 21, 2] 예) 말씀이 중국과 달라

        loss, ps, hs, xs = feedforward(inputs, targets, hprev)

        dU, dW, dV, hprev = backward(ps, hs, xs)

        # Update weights and biases using gradient descent
        W -= learning_rate * dW
        U -= learning_rate * dU
        V -= learning_rate * dV

        # count += 1
        # print('loop count = ',count)

    if epoch % 100 == 0:
        print(f'epoch {epoch}, loss: {loss}')

while 1:
    try:
        user_input = input("input: ")
        if user_input == 'break':
            break
        response = predict(user_input,40)
        print(response)
    except:
        print('Uh oh try again!')