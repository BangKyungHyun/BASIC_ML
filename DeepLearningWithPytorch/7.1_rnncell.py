################################################################################
# 2024.6.6
# 라이브러리 호출
################################################################################

import torch
# (base) C:\Users\bangkh21>pip install torchtext==0.11.1
import torchtext
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime

################################################################################
# 데이터 전 처리
################################################################################

start = time.time()

# torchtext.legacy.data는 제공하는 Field는 데이터 전처리를 위해 사용됨
# lower : 대표자를 모두 소문자로 변경, 기본값은 false
# fixed_length : 고정된 길이의 데이터를 얻을 수 있음. 여기에서는 데이터의 길이를 200으로
#                고정했으며 200보다 짧으면 패딩작업을 통해 200으로 맞춤
# batch_first  : 신경망에 입력되는 텐서의 첫번째 차원 값이 배치크기로 되도록 함
#                기본값은 false임. 모델의 네트워크로 입력되는 데이터는 [시퀀스 길이, 배치 크기,
#                은닉층의 뉴런 개수]([seq_len, batch_size, hidden_size])의 형태임

TEXT = torchtext.legacy.data.Field(lower=True, fix_length=200, batch_first=False)
LABEL = torchtext.legacy.data.Field(sequential=False)

################################################################################
# 데이터 셋 준비
################################################################################

# datasets.IMDB를 사용하여 IMDB데이터셋을 내려 받음

from torchtext.legacy import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# 훈련 데이터셋의 첫번째 examples[0] 출력
print(vars(train_data.examples[0]))

# {'text': ['bromwell', 'high', 'is', 'a', 'cartoon', 'comedy.', 'it', 'ran', 'at', 'the', 'same', 'time', 'as', 'some', 'other', 'programs', 'about', 'school', 'life,',
# 'such', 'as', '"teachers".', 'my', '35', 'years', 'in', 'the', 'teaching', 'profession', 'lead', 'me', 'to', 'believe', 'that', 'bromwell', "high's", 'satire', 'is',
# 'much', 'closer', 'to', 'reality', 'than', 'is', '"teachers".', 'the', 'scramble', 'to', 'survive', 'financially,', 'the', 'insightful', 'students', 'who', 'can', 'see',
# 'right', 'through', 'their', 'pathetic', "teachers'", 'pomp,', 'the', 'pettiness', 'of', 'the', 'whole', 'situation,', 'all', 'remind', 'me', 'of', 'the', 'schools', 'i',
# 'knew', 'and', 'their', 'students.', 'when', 'i', 'saw', 'the', 'episode', 'in', 'which', 'a', 'student', 'repeatedly', 'tried', 'to', 'burn', 'down', 'the', 'school,', 'i',
# 'immediately', 'recalled', '.........', 'at', '..........', 'high.', 'a', 'classic', 'line:', 'inspector:', "i'm", 'here', 'to', 'sack', 'one', 'of', 'your', 'teachers.', 'student:',
# 'welcome', 'to', 'bromwell', 'high.', 'i', 'expect', 'that', 'many', 'adults', 'of', 'my', 'age', 'think', 'that', 'bromwell', 'high', 'is', 'far', 'fetched.', 'what', 'a', 'pity',
# 'that', 'it', "isn't!"], 'label': 'pos'}

################################################################################
# 데이터 전 처리 적용
################################################################################

import string

for example in train_data.examples:
    # 소문자로 변경
    text = [x.lower() for x in vars(example)['text']]

    # "<br"을 공백으로 변경(개행)
    text = [x.replace("<br", "") for x in text]

    # 구둣점 제거
    text = [''.join(c for c in s if c not in string.punctuation) for s in text]

    # 공백 제거
    text = [s for s in text if s]

    vars(example)['text'] = text

################################################################################
# 훈련과 검증 데이터셋 분리
################################################################################

import random

# split()을 이용하여 훈련 데이터셋을 훈련과 검증 용도로 분리
# random_state : 데이터 분할 시 데이터가 임의로 섞인 상태에서 분할. 이때 시드값을 사용하면
#                동일한 코드를 여러번 수행하더라도 동일한 값의 데이터를 반환
# split_rate : 데이터 분할 정도를 의미. 8:2로 분리함
train_data, valid_data = train_data.split(random_state = random.seed(0), split_ratio=0.8)

################################################################################
# 데이터셋 갯수 확인
################################################################################

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# Number of training examples: 20000
# Number of validation examples: 5000
# Number of testing examples: 25000

################################################################################
# 단어 집합 만들기
################################################################################

# 단어 집합이란 IMDB 데이터셋에 포함된 단어들을 이용하여 하나의 딕셔너리와 같은 집합을 만듬
# 단어 집합을 만들때는 단어들의 중복은 제거된 상태에서 제거

# 단어 집합 생성은 TEXT.build_vocab()을 이용
# train_data : 훈련 데이터셋
# max_size : 단어 집합의 크기로 단어 집합에 포함되는 어휘 수를 의미
# min_freq : 훈련 데이터셋에서 특정 단어의 최소 등장 횟수를 의미
#            min_freq=10으로 설정했기 때문에 훈련 데이터셋에서 특정 단어가 최소 10번 이상
#            등장한 것만 집합에 추가한다는 의미
# vectors :  임베딩 벡터를 지정할 수 있음
#            임베딩 벅터는 워드 임베딩의 결과로 나온 벡터
#            사전 학습된 임베딩으로는 word2vec, glove등이 있음
#            파이토치에도 nn.embedding()을 통해 단어를
#            랜덤한 숫자 값으로 변환한 후 가중치를 학습하는 방법 제공

TEXT.build_vocab(train_data, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train_data)

# print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
# print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

# Unique tokens in TEXT vocabulary: 10002
# Unique tokens in LABEL vocabulary: 3

################################################################################
# 테스트 데이터셋의 단어 집합 확인
################################################################################

# LABEL.vocab.stoi을 통해 현재 단어 집합의 단어와 그것에 부여된 고유정수(인덱스)를 확인
# <unk>는 사전에 없는 단어를 의미

# print(TEXT.vocab.stoi)

# defaultdict(<bound method Vocab._default_unk_index of <torchtext.legacy.vocab.Vocab object at 0x00000285A6584280>>, {'<unk>': 0, '<pad>': 1, 'the': 2, 'and': 3, 'a': 4, 'of': 5, 'to': 6, 'is': 7, 'in':
#  ......................................
# 9990, 'y': 9991, 'zoo': 9992, '5th': 9993, '65': 9994, 'abbot': 9995, 'abbott': 9996, 'acquire': 9997, 'actuality': 9998, 'adapting': 9999, 'adelaide': 10000, 'akira': 10001})
# defaultdict(<bound method Vocab._default_unk_index of <torchtext.legacy.vocab.Vocab object at 0x00000285CA64C280>>, {'<unk>': 0, 'pos': 1, 'neg': 2})
# len(TEXT.vocab.stoi) =  10002

print(LABEL.vocab.stoi)
# defaultdict(<bound method Vocab._default_unk_index of <torchtext.vocab.Vocab object at 0x000002E1341FAE20>>, {'<unk>': 0, 'pos': 1, 'neg': 2})

################################################################################
# 데이터셋 메모리로 가져오기
################################################################################

# 전처리가 완료되었기 때문에 BucketIterator()를 이용하여 데이터셋을 메모리로 가져옴

BATCH_SIZE = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 각 단어를 100 차원으로 조정(임베딩 계층을 통과한 후 각 벡터의 크기)
embeding_dim = 100

# 은닉층의 유닛 갯수를 지정
# 일반적으로 계층(layer)의 유닛갯수를 늘리는 것보다 계층 자체에 대한 갯수를 늘리는 것이 성능을 위해서는 더 좋음
hidden_size = 300

# BucketIterator()는 데이터 로더와 쓰임새가 같음. 즉, 배치 크기 단위로 값을 차례대로 꺼내어
# 메모리로 가져오고 싶을 때 사용
# 특히 Field에서 fix_length를 사용하지 않는다면 BucketIterator()에서 데이터 길이를 조정할 수 있음
# BucketIterator()는 비슷한 길이의 데이터를 한 배치에 할당하여 패딩을 최소화 시팀
# 1번째 파라미터 : 배치 크기 단위로 데이터를 가져올 데이터셋
# 2번째 파라미터 : 한번에 가져올 데이터 크기(배치 크기)
# 3번째 파라미터 : 어떤 장치(CPU OR GPU)를 사용할 지 결정

train_iterator, valid_iterator, test_iterator = \
    torchtext.legacy.data.BucketIterator.splits((train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,device = device)

################################################################################
# 워드 임베딩 및 RNN 셀 정의
################################################################################

# 앞에서 단어 집합을 만드는 과정에서 vectors=none으로 설정했기 때문에 임베딩 부분에 대해 정의하지 않음
# 이번 예제에서는 nn.Embedding()을 이용하여 임베딩 처리를 진행함

class RNNCell_Encoder(nn.Module):

    # hidden_size = 300
    def __init__(self, input_dim, hidden_size):
        super(RNNCell_Encoder, self).__init__()

        # RNN 셀 구현을 위한 구문
        # input_dim : 훈련 데이터의 특성 갯수로(배치, 입력 데이터 컬럼 갯수/특성 갯수(batch, input_size)) 형태를 갖음
        # hidden_size : 은닉층의 뉴런(유닛) 갯수로 (배치, 은닉층의 뉴런 갯수(batch, hidden_size)) 형태를 갖음

        # print('RNNCell_Encoder(nn.Module),input_dim, hidden_size =', input_dim, hidden_size)
        # RNNCell_Encoder(nn.Module), input_dim, hidden_size = 100   300
        self.rnn = nn.RNNCell(input_dim, hidden_size)

    # inputs는 입력 시퀀스로(시퀀스 길이, 배치, 임베딩(seq, batch, embedding))의 형태를 갖음
    def forward(self, inputs):
        # 배치를 가져옴
        # print('RNNCell_Encoder(nn.Module), inputs, inputs.shape[1] =', inputs.shape,inputs.size, inputs.shape[1])
        # RNNCell_Encoder(nn.Module), inputs, inputs.shape[1] = torch.Size([200, 64, 100]) <built-in method size of Tensor object at 0x0000021834003770> 64

        # inputs.shape[1] = 64
        bz = inputs.shape[1]

        # 배치와 은닉층 뉴런의 크기를 0으로 초기화
        ht = torch.zeros((bz, hidden_size)).to(device)  # 현재상태(h_t)
        # print('RNNCell_Encoder(nn.Module), ht.shape, ht.size =', ht.shape, ht.size)
        # RNNCell_Encoder(nn.Module), ht.shape, ht.size = torch.Size([64, 300]) <built-in method size of Tensor object at 0x00000208C963D040>

        for word in inputs:  # word : 현재 상태(x_t)
            # 재귀적으로 발생하는 상태 값을 처리하기 위한 구문 
            # ht : 현재의 상태를 의미하는 것으로 ht을 뜻함
            # word : 현재의 입력 벡터로 배치, 입력 데이터 컬럼 갯수(batch, input_size))의 형태를 가짐
            # ht : 이전 상태를 의미하는 것으로 h(t-1)을 뜻함
            ht = self.rnn(word, ht)   # ht : 이전 상태(h_t-1)
        return ht

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        print('len(TEXT.vocab.stoi) = ', len(TEXT.vocab.stoi))
        # len(TEXT.vocab.stoi) = 10002

        # 임베딩 처리를 위한 구문
        # len(TEXT.vocab.stoi) : 임베딩을 할 단어 수(단어 집합의 크기)
        # embeding_dim : 임베딩할 벡터의 차원 ==> 100
        self.em = nn.Embedding(len(TEXT.vocab.stoi), embeding_dim)

        self.rnn = RNNCell_Encoder(embeding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.em(x)
        x = self.rnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

################################################################################
# 옵티마이저와 손실 함수 정의
################################################################################

# model이라는 이름으로 모델을 객체화 
model = Net()
model.to(device)

# CrossEntropyLoss()는 다중 분류에 사용하며 nn.LogSoftMax와 nn.NULLoss 연산의 조합으로 구성
# nn.LogSoftMax는 모델 네트워크의 마지작 계층에서 얻은 결과값을 확률로 해설하기 위해 소프트맥스
# 함수의 결과에 로그를 취함. 
# nn.NULLLoss 다중분류에 사용함. 신경망에서 로그 확률 값을 얻으려면 마지막에 LogSoftMax를 추가해야 함 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

################################################################################
# 모델 학습을 위한 함수 정의
################################################################################

def training(epoch, model, trainloader, validloader):

    correct = 0
    total = 0
    running_loss = 0

    model.train()

    for b in trainloader:
        # trainloader에서 text와 label를 꺼내 옴
        x, y = b.text, b.label
        # 꺼내 온 데이터가 cpu을 사용할 수 있도록 장치 지정, 반드시 모델과 같은 장치를 사용하도록 지정해야 함
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        # cross entropyLoss 손실함수를 이용하여 오차 계산
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

    # 누적된 오차를 전체 데이터셋으로 나누어서 에포크 단계마다 오차를 구함
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct / total

    valid_correct = 0
    valid_total = 0
    valid_running_loss = 0

    model.eval()

    with torch.no_grad():
        for b in validloader:
            x, y = b.text, b.label
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            y_pred = torch.argmax(y_pred, dim=1)
            valid_correct += (y_pred == y).sum().item()
            valid_total += y.size(0)
            valid_running_loss += loss.item()

    epoch_valid_loss = valid_running_loss / len(validloader.dataset)
    epoch_valid_acc = valid_correct / valid_total

    # 훈련이 진행될 때 에포크마다 정확도와 오차를 출력

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    print(nowDatetime,
          'epoch: ', epoch,
          'loss： ', round(epoch_loss, 7),
          'accuracy:', round(epoch_acc, 7),
          'valid_loss： ', round(epoch_valid_loss, 7),
          'valid_accuracy:', round(epoch_valid_acc, 7)
          )
    return epoch_loss, epoch_acc, epoch_valid_loss, epoch_valid_acc

################################################################################
# 모델 학습
################################################################################

epochs = 5
train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

for epoch in range(epochs):

    epoch_loss, epoch_acc, epoch_valid_loss, epoch_valid_acc = \
        training(epoch, model,train_iterator, valid_iterator)

    # 훈련 데이터셋을 모델에 적용했을 때의 오차
    train_loss.append(epoch_loss)

    # 훈련 데이터셋을 모델에 적용했을 때의 정확도
    train_acc.append(epoch_acc)

    # 검증 데이터셋을 모델에 적용했을 때의 오차
    valid_loss.append(epoch_valid_loss)

    # 검증 데이터셋을 모델에 적용했을 때의 정확도
    valid_acc.append(epoch_valid_acc)

end = time.time()
print('working time = ',(end-start)/60)

################################################################################
# 모델 예측 함수 정의
################################################################################

def evaluate(epoch, model, testloader):

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()
    with torch.no_grad():
        for b in testloader:
            x, y = b.text, b.label
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / test_total

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    print(nowDatetime,
          'epoch: ', epoch,
          'test_loss： ', round(epoch_test_loss, 7),
          'test_accuracy:', round(epoch_test_acc, 7)
          )
    return epoch_test_loss, epoch_test_acc

################################################################################
# 모델 예측 결과 확인
################################################################################

epochs = 1
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_test_loss, epoch_test_acc = evaluate(epoch, model, test_iterator)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

end = time.time()
print('working time = ', (end-start)/60)